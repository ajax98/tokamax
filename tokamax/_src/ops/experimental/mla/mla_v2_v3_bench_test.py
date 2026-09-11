# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Side-by-side MLA v2 vs v3 benchmark over every arg-spec workload.

Six settings x two implementations. Not a correctness test -- numerics are
covered by `mla/v2_v3_op_test.py` and the per-version kernel tests. This exists
to reproduce the comparison at each workload's best known block sizes, in one
place, without the external harness scripts.

Methodology, all three parts of which matter by 2x or more:

  * **The v3 KV layout conversion is outside the timed region.** v3 wants the
    cache in SEQ_ALONG_LANE layout; converting per call costs ~10 ms on the
    755 MB decode cache, 15x the attention itself. A deployed system holds the
    cache in v3 layout permanently, so converting here would measure migration
    cost rather than kernel cost.
  * **`cache_kv` is donated and the result chained into the next call.**
    Otherwise XLA copies the whole cache every iteration (the kernel's
    `input_output_aliases` wants to write in place while the caller still owns
    the buffer), and whether that copy overlaps the kernel silently changes the
    answer by 2x. Chaining is also what a serving loop does.
  * **Wallclock is reported, device time is what was tuned on.** Host variance
    is ~0.05 ms here, which is enough to invert small differences -- it once
    made a real 6.3% kernel win look like noise. Treat the printed numbers as
    indicative; use xprof for anything finer.

Provenance of the block sizes, which is *not* uniform:

  TUNED_DONATED   swept under the current (donated) harness. Trustworthy.
  TUNED_LEGACY    swept before the donation fix. The metric those sweeps ranked
                  on was polluted by the per-iteration cache copy, and it
                  demonstrably mis-ranked elsewhere -- it picked
                  num_kv_pages_per_block=1 for v3 decode where 8 was 1.9x
                  better. Treat as a starting point, not an optimum.
  HEURISTIC       never swept. Reasonable values only.

Run with `-s` to see the summary table.
"""

import functools
import gc
import time

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.mla import test_base
from tokamax._src.ops.experimental.mla.v2 import kernel as kernel_v2
from tokamax._src.ops.experimental.mla.v3 import mla_wrapper
from tokamax._src.ops.experimental.mla.v3 import utils as utils_v3

jax.config.update("jax_numpy_dtype_promotion", "standard")

_BF16 = jnp.bfloat16
_FP8 = jnp.float8_e4m3fn
_MB = 1024 * 1024

# v3 flags that are mechanism-general wins rather than shape-specific: both
# reduce work unconditionally and were measured at -6.3% and -3.0% on decode.
# `fast_mask` only engages when a query block holds one token, so it is
# harmless (inert) on the prefill settings.
_V3_GENERAL = dict(
    fast_mask=True,
    disable_bounds_checks=True,
    merge_kv_dma=True,
)

# `kv_slack_pad_lanes=128` makes the stitch stride an odd multiple of 128.
# Measured -10.5% on decode; a power-of-two stride aliases onto VMEM banks and
# cost +49%. Applied only where measured -- the prefill settings have a
# different `bkv_sz` and therefore a different stride, so it needs its own
# sweep there before being assumed good.

SETTINGS = {
    # ---- the three original arg-specs -------------------------------------
    "decode_bf16": dict(
        seq_lens=[(1, 8192)] * 3,
        page_size=256,
        q_dtype=_BF16,
        kv_dtype=_BF16,
        provenance="HEURISTIC",
        # Only 3 sequences, and v2 requires decode_batch_size to divide the
        # decode band, so 1 is the only legal batching here.
        v2=dict(
            num_kv_pages_per_block=8, num_queries_per_block=1,
            decode_batch_size=1, mixed_q_split=1, vmem_limit_bytes=64 * _MB,
        ),
        v3=dict(
            num_kv_pages_per_block=8, num_queries_per_block=1,
            batch_size=1, n_buffer=2, vmem_limit_bytes=64 * _MB, **_V3_GENERAL,
        ),
    ),
    "prefill_bf16": dict(
        seq_lens=[(2048, 2048)] * 2,
        page_size=256,
        q_dtype=_BF16,
        kv_dtype=_BF16,
        provenance="HEURISTIC",
        v2=dict(
            num_kv_pages_per_block=8, num_queries_per_block=16,
            decode_batch_size=1, mixed_q_split=1, vmem_limit_bytes=64 * _MB,
        ),
        v3=dict(
            num_kv_pages_per_block=8, num_queries_per_block=16,
            batch_size=1, n_buffer=2, vmem_limit_bytes=64 * _MB, **_V3_GENERAL,
        ),
    ),
    "decode_f8": dict(
        seq_lens=[(1, 8192)] * 128,
        page_size=256,
        q_dtype=_FP8,
        kv_dtype=_FP8,
        provenance="HEURISTIC",
        # Mirrors the swept optimum for the near-identical
        # `decode_f8_kv9216` at page_size=256.
        v2=dict(
            num_kv_pages_per_block=8, num_queries_per_block=1,
            decode_batch_size=8, mixed_q_split=1, vmem_limit_bytes=64 * _MB,
        ),
        v3=dict(
            num_kv_pages_per_block=8, num_queries_per_block=1,
            batch_size=8, n_buffer=2, vmem_limit_bytes=64 * _MB, **_V3_GENERAL,
        ),
    ),
    # ---- the three FP8 comparison specs -----------------------------------
    "chunked_prefill_f8_kv1024": dict(
        seq_lens=[(256, 1024)],
        page_size=256,
        q_dtype=_FP8,
        kv_dtype=_FP8,
        provenance="TUNED_LEGACY",
        v2=dict(
            num_kv_pages_per_block=4, num_queries_per_block=16,
            decode_batch_size=1, mixed_q_split=16, vmem_limit_bytes=48 * _MB,
        ),
        v3=dict(
            num_kv_pages_per_block=4, num_queries_per_block=16,
            batch_size=1, n_buffer=3, vmem_limit_bytes=64 * _MB, **_V3_GENERAL,
        ),
    ),
    "chunked_prefill_f8_kv8192": dict(
        seq_lens=[(256, 8192)],
        page_size=256,
        q_dtype=_FP8,
        kv_dtype=_FP8,
        provenance="TUNED_LEGACY",
        v2=dict(
            num_kv_pages_per_block=32, num_queries_per_block=16,
            decode_batch_size=1, mixed_q_split=16, vmem_limit_bytes=64 * _MB,
        ),
        v3=dict(
            num_kv_pages_per_block=16, num_queries_per_block=16,
            batch_size=1, n_buffer=2, vmem_limit_bytes=64 * _MB, **_V3_GENERAL,
        ),
    ),
    "decode_f8_kv9216_p1024": dict(
        seq_lens=[(1, 9216)] * 128,
        page_size=1024,
        q_dtype=_FP8,
        kv_dtype=_FP8,
        provenance="TUNED_DONATED",
        # Device time at this configuration: v2 0.4033 ms kernel / 0.4612
        # total; v3 0.5445 / 0.6628 -- 1.35x kernel, 1.44x end to end.
        v2=dict(
            num_kv_pages_per_block=3, num_queries_per_block=1,
            decode_batch_size=8, mixed_q_split=1, vmem_limit_bytes=64 * _MB,
        ),
        v3=dict(
            num_kv_pages_per_block=3, num_queries_per_block=1,
            batch_size=4, n_buffer=2, vmem_limit_bytes=64 * _MB,
            kv_slack_pad_lanes=128, **_V3_GENERAL,
        ),
    ),
}

_ITERATIONS = 10
_WARMUP = 2
# Loose: catches a lost flag or a broken config, not noise. `prefill_bf16` is
# the largest workload here (4096 query tokens), hence the headroom.
_BUDGET_MS = 50.0

_RESULTS = {}


class MlaV2V3BenchTest(parameterized.TestCase):
  """Times v2 and v3 on each workload at its best known block sizes."""

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()
    gc.collect()

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    if not _RESULTS:
      return
    print(f"\n{'=' * 78}")
    print("MLA v2 vs v3 -- median wallclock (ms), best known block sizes")
    print(f"{'=' * 78}")
    print(f"{'setting':<30} {'page':>5} {'v2':>9} {'v3':>9} {'v3/v2':>7}  tuning")
    for name, cfg in SETTINGS.items():
      v2 = _RESULTS.get((name, "v2"))
      v3 = _RESULTS.get((name, "v3"))
      ratio = f"{v3 / v2:.2f}x" if (v2 and v3) else "-"
      print(
          f"{name:<30} {cfg['page_size']:>5} "
          f"{v2 if v2 else float('nan'):>9.4f} "
          f"{v3 if v3 else float('nan'):>9.4f} {ratio:>7}  {cfg['provenance']}"
      )
    print(
        "\nWallclock; host variance ~0.05 ms. TUNED_LEGACY block sizes were"
        "\nswept before the donation fix and are likely not optimal."
    )

  def _build(self, cfg):
    return test_base.generate_mla_inputs(
        cfg["seq_lens"],
        128,  # num_heads
        512,  # lkv_dim
        64,  # r_dim
        cfg["page_size"],
        cfg["q_dtype"],
        cfg["kv_dtype"],
        128,  # num_pages floor
        rng=np.random.default_rng(1234),
    )

  def _time(self, fn, cache_kv):
    for _ in range(_WARMUP):
      out, cache_kv = fn(cache_kv)
    jax.block_until_ready((out, cache_kv))
    times = []
    for _ in range(_ITERATIONS):
      t0 = time.perf_counter()
      out, cache_kv = fn(cache_kv)
      jax.block_until_ready((out, cache_kv))
      times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times)), min(times)

  @parameterized.named_parameters(
      *[(f"{n}__{i}", n, i) for n in SETTINGS for i in ("v2", "v3")]
  )
  def test_bench(self, setting, impl):
    if not jax.devices() or jax.devices()[0].platform != "tpu":
      self.skipTest("Expect TPU")

    cfg = SETTINGS[setting]
    (
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
        kv_lens, page_indices, cu_q_lens, distribution,
    ) = self._build(cfg)

    # Both kernels take `ql_nope` head-major; `generate_mla_inputs` emits
    # token-major.
    ql_nope = jnp.transpose(ql_nope, (1, 0, 2))

    if impl == "v3":
      packing = test_base.get_dtype_packing(cfg["kv_dtype"])
      cache_kv = utils_v3.transpose_kv_cache_to_v3(cache_kv, packing)

      @functools.partial(jax.jit, donate_argnames=("cache_kv",))
      def run(cache_kv):
        return mla_wrapper.mla_ragged_paged_attention(
            ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
            kv_lens, page_indices, cu_q_lens, distribution, **cfg["v3"],
        )
    else:

      @functools.partial(jax.jit, donate_argnames=("cache_kv",))
      def run(cache_kv):
        return kernel_v2.mla_ragged_paged_attention(
            ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
            kv_lens, page_indices, cu_q_lens, distribution, **cfg["v2"],
        )

    median_ms, min_ms = self._time(run, cache_kv)
    _RESULTS[(setting, impl)] = median_ms

    logging.info(
        "%s / %s: median %.4f ms, min %.4f ms [%s]",
        setting, impl, median_ms, min_ms, cfg["provenance"],
    )
    print(
        f"{setting:<30} {impl:<3} median {median_ms:8.4f} ms  "
        f"min {min_ms:8.4f} ms  [{cfg['provenance']}]"
    )
    self.assertLess(median_ms, _BUDGET_MS)


if __name__ == "__main__":
  absltest.main()

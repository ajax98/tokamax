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

"""Times MLA v2 against v3 on seven workloads, each at its tuned block sizes.

Both sides are swept, so these are best-vs-best. Reported numbers are
wallclock; the block sizes were chosen on *device* occupancy, which inverted
the result on four of the seven specs -- host overhead is over half the
measurement on the smaller ones. Treat wallclock as a regression guard and
xprof as the source of truth.

Device time per iteration (tpu7x, jax 0.11.1):

    spec                        v2_tot   v3_tot   totx   v2_ker   v3_ker  kerx
    prefill_bf16                9.7253   2.8922  3.363   8.0844   2.6887 3.007
    chunked_prefill_f8_kv1024   0.1756   0.1332  1.318   0.0941   0.0987 0.953
    decode_f8                   0.4800   0.4720  1.017   0.4164   0.4112 1.013
    chunked_prefill_f8_kv8192   0.6106   0.6269  0.974   0.5290   0.5719 0.925
    decode_f8_kv9216            0.4896   0.5179  0.945   0.4265   0.4537 0.940
    decode_f8_kv9216_p1024      0.4652   0.5063  0.919   0.4023   0.4267 0.943
    decode_bf16                 0.0533   0.0728  0.732   0.0280   0.0274 1.019

v3 wins prefill on causal block skipping: `schedule.q_loop` emits tasks only
for the k-blocks a q-block can reach, which halves the work at q_len == kv_len
and is worth nothing once history exceeds one bkv_sz. Where it wins end to end
without winning the kernel (kv1024) the margin is lower non-kernel overhead.
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
_VMEM = 64 * 1024 * 1024


def _s(seq_lens, page_size, dtype, v2, v3):
  return dict(
      seq_lens=seq_lens,
      page_size=page_size,
      q_dtype=dtype,
      kv_dtype=dtype,
      v2={**v2, "vmem_limit_bytes": _VMEM},
      v3={**v3, "n_buffer": 2, "vmem_limit_bytes": _VMEM},
  )


SETTINGS = {
    "decode_bf16": _s(
        [(1, 8192)] * 3, 256, _BF16,
        dict(num_kv_pages_per_block=8, num_queries_per_block=1,
             decode_batch_size=1, mixed_q_split=1),
        dict(num_kv_pages_per_block=8, num_queries_per_block=1, batch_size=4),
    ),
    "prefill_bf16": _s(
        [(2048, 2048)] * 2, 256, _BF16,
        dict(num_kv_pages_per_block=8, num_queries_per_block=16,
             decode_batch_size=1, mixed_q_split=4),
        dict(num_kv_pages_per_block=2, num_queries_per_block=32, batch_size=1,
             q_split=8),
    ),
    "decode_f8": _s(
        [(1, 8192)] * 128, 256, _FP8,
        dict(num_kv_pages_per_block=8, num_queries_per_block=1,
             decode_batch_size=8, mixed_q_split=1),
        # kv=32 makes bkv_sz == kv_len: one k-block per sequence, so the
        # schedule emits 128 tasks rather than 512.
        dict(num_kv_pages_per_block=32, num_queries_per_block=1, batch_size=2),
    ),
    "chunked_prefill_f8_kv1024": _s(
        [(256, 1024)], 256, _FP8,
        dict(num_kv_pages_per_block=4, num_queries_per_block=32,
             decode_batch_size=1, mixed_q_split=4),
        dict(num_kv_pages_per_block=4, num_queries_per_block=32, batch_size=1,
             q_split=8, p_same_dtype_as_v=True),
    ),
    "chunked_prefill_f8_kv8192": _s(
        [(256, 8192)], 256, _FP8,
        dict(num_kv_pages_per_block=8, num_queries_per_block=32,
             decode_batch_size=1, mixed_q_split=4),
        # q_split=16 is what makes kv=8 fit: the live score tile is
        # [bq_sz * heads, bkv_sz] and q_split divides the first factor.
        # p_same_dtype_as_v is -16.1% here and +9.4% on decode.
        dict(num_kv_pages_per_block=8, num_queries_per_block=32, batch_size=1,
             q_split=16, p_same_dtype_as_v=True),
    ),
    "decode_f8_kv9216": _s(
        [(1, 9216)] * 128, 256, _FP8,
        dict(num_kv_pages_per_block=12, num_queries_per_block=1,
             decode_batch_size=8, mixed_q_split=1),
        dict(num_kv_pages_per_block=36, num_queries_per_block=1, batch_size=2),
    ),
    "decode_f8_kv9216_p1024": _s(
        [(1, 9216)] * 128, 1024, _FP8,
        dict(num_kv_pages_per_block=3, num_queries_per_block=1,
             decode_batch_size=8, mixed_q_split=1),
        # pad=128 makes the stitch stride an odd multiple of 128; a
        # power-of-two stride aliases onto VMEM banks. -10.5% here, and inert
        # at the whole-sequence block sizes the other decode specs use.
        dict(num_kv_pages_per_block=3, num_queries_per_block=1, batch_size=4,
             kv_slack_pad_lanes=128),
    ),
}

_ITERATIONS = 10
_WARMUP = 2
_BUDGET_MS = 50.0  # catches a broken config, not noise
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
    print(f"\n{'=' * 66}")
    print("MLA v2 vs v3 -- median wallclock (ms), tuned block sizes")
    print(f"{'=' * 66}")
    print(f"{'setting':<30} {'page':>5} {'v2':>9} {'v3':>9} {'v3/v2':>7}")
    for name, cfg in SETTINGS.items():
      v2 = _RESULTS.get((name, "v2"))
      v3 = _RESULTS.get((name, "v3"))
      ratio = f"{v3 / v2:.2f}x" if (v2 and v3) else "-"
      print(
          f"{name:<30} {cfg['page_size']:>5} "
          f"{v2 if v2 else float('nan'):>9.4f} "
          f"{v3 if v3 else float('nan'):>9.4f} {ratio:>7}"
      )
    print("\nWallclock; host variance ~0.05 ms.")

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
        "%s / %s: median %.4f ms, min %.4f ms", setting, impl, median_ms, min_ms
    )
    print(
        f"{setting:<30} {impl:<3} median {median_ms:8.4f} ms  "
        f"min {min_ms:8.4f} ms"
    )
    self.assertLess(median_ms, _BUDGET_MS)


if __name__ == "__main__":
  absltest.main()

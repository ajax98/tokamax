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

# The mechanism-general v3 wins -- flat_q (-18% total / -13.9% kernel),
# disable_bounds_checks (-6.3%), fuse_qk (-5.6%), fast_mask (-3.4%),
# gate_stitch (-2.7%), static_kv_dst (-1.8% schedule), merge_kv_dma -- are no
# longer flags. Each measured as a win on decode and neutral-or-better on
# prefill, so they became unconditional kernel behaviour and there is nothing
# left to set here. `_V3_GENERAL` is retained as an empty base so the
# per-setting dicts below keep their shape.
_V3_GENERAL = {}

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
        provenance="TUNED_DONATED",
        # Device: v2 0.0533 total / 0.0281 kernel against v3 0.0539 / 0.0274 --
        # parity, v3 +1.1% end to end and -2.5% on the kernel. The smallest
        # workload in this file by two orders of magnitude, so fixed per-call
        # overhead dominates and neither side has much room.
        #
        # Only 3 sequences, and v2 requires decode_batch_size to divide the
        # decode band, so 1 is the only legal batching for v2. v3's batch_size
        # is unrelated (consecutive schedule tasks, not sequences) and 4 wins.
        v2=dict(
            num_kv_pages_per_block=8, num_queries_per_block=1,
            decode_batch_size=1, mixed_q_split=1, vmem_limit_bytes=64 * _MB,
        ),
        v3=dict(
            num_kv_pages_per_block=8, num_queries_per_block=1,
            batch_size=4, n_buffer=2,
            vmem_limit_bytes=64 * _MB, **_V3_GENERAL,
        ),
    ),
    "prefill_bf16": dict(
        seq_lens=[(2048, 2048)] * 2,
        page_size=256,
        q_dtype=_BF16,
        kv_dtype=_BF16,
        provenance="TUNED_DONATED",
        # Swept on both sides: 18 v2 configs and ~35 v3. Device time at these
        # settings is v2 9.6454 ms total / 8.0041 kernel against v3 2.7806 /
        # 2.6138 -- v3 was 3.47x faster end to end, 3.06x on the kernel.
        #
        # NOTE: 3.47x is no longer reachable from this file. Its last step was
        # `gate_stitch=False`, and the stitch gate is now unconditional, so the
        # attainable figure here is the 3.37x below. The 3.47x is kept as the
        # record of what the shape can do, not as a current expectation.
        #
        # Progression, all device: 2.66x with block sizes alone, 2.95x adding
        # `q_split=8` + the two-step PV deferral (-9.7%), 3.29x once `ql_nope`
        # is donated in the harness (that was a 512 MB identity copy charged
        # only to v3, see run_mla_profile.py), 3.33x with
        # `kv_slack_pad_lanes=0` -- and 3.37x with the since-removed `bands`, whose
        # distribution is [0,0,n], so only the MIXED band is ever non-empty and
        # saying so drops two lax.cond branches (-1.5% total, kernel
        # unchanged), and 3.47x with `gate_stitch=False`.
        #
        # Also measured here and *not* adopted: `fuse_qk=False` costs +136%, so
        # unlike on decode it is load-bearing; `merge_kv_dma`, `flat_q`,
        # `static_kv_dst` and `disable_bounds_checks` are all within noise on
        # this shape; a compact KV cache is exactly zero (the MXU quantizes the
        # contraction to 128-element tiles, and 640 and 576 both round to 5);
        # `k_split` costs +22%.
        #
        # Why: `schedule.q_loop` only emits tasks for the k-blocks a q-block
        # can causally reach, so at bkv_sz=512 it runs 160 of the 256 tasks in
        # the rectangle. v2's optimum puts the whole 2048-token sequence in one
        # k-block, where there is nothing to skip -- and it gets *slower* with
        # smaller blocks (11.51 ms at kv=2 vs 10.85 at kv=8), so it is not
        # exploiting the granularity either way.
        #
        # `q_split=8` + `two_step` is worth a further -9.7% total / -11.4%
        # kernel (3.6230 -> 3.2729), taking the ratio from 2.66x to 2.95x.
        # `bq_c_sz` had been pinned to `bq_sz`, so this path was unreachable.
        # It works here and not on decode for one reason: n_q is 4096 rows
        # (32 MXU tiles) at bq_sz=32, so splitting eight ways leaves four whole
        # tiles per chunk, where decode's n_q of 128 is a single tile.
        # q_split 16 and 32 collapse (4.43 and 6.34) at exactly the point
        # chunks stop spanning multiple tiles.
        #
        # Deliberately no numeric knobs here. `narrow_scores +
        # narrow_softmax + p_same_dtype_as_v` measured 3.4769 ms, a further 4%,
        # and is *wrong* on this shape: 17.99% of elements outside
        # atol=rtol=2e-2, maxdiff 0.2188, against 0 for every block-size-only
        # config. It passes the decode knob tests because those are fp8 at
        # atol=0.1/rtol=0.2; bf16 at 2e-2 has no room for a bf16 `s - m` and
        # `exp`. See `check_prefill_bf16.py`.
        #
        # v2 is close to flat across its space (9.69 - 10.35 over the five
        # sensible configs; `mixed_q_split` 1/4/8 are within 0.8% of each
        # other), so this is not an undertuned baseline. It degrades badly
        # off-optimum -- q=4 costs 24.9 ms and kv=16 costs 12.9 -- but nothing
        # beats ~9.7.
        #
        # v3 prefers a *small* KV block here: kv=2 (bkv_sz 512) against the
        # kv=8 the heuristic used, worth 4.65 -> 4.32 ms, and q=32 over q=16.
        # q=64 and batch_size=2 both exceed VMEM at this q.
        v2=dict(
            num_kv_pages_per_block=8, num_queries_per_block=16,
            decode_batch_size=1, mixed_q_split=4, vmem_limit_bytes=64 * _MB,
        ),
        v3=dict(
            num_kv_pages_per_block=2, num_queries_per_block=32,
            batch_size=1, n_buffer=2, q_split=8,
            vmem_limit_bytes=64 * _MB,
            # Overrides the Config default of 128, which is decode's value.
            # Prefill wants 0, i.e. a 1024-lane KV buffer -- a power of two,
            # which is precisely the stride that cost decode +49%. The aliasing
            # that dominates a 5248-lane decode buffer evidently does not bind
            # here. Worth -1.2%.
            # Overrides the Config default of 128, tuned on decode:
            #   kv_slack_pad_lanes 128 -> 0   (-1.2%)
            #
            # This setting also measured -3.1% with `gate_stitch=False`, which
            # is no longer expressible: the stitch gate is now unconditional.
            # Whole-prompt prefill has q_len == kv_len, so every KV token is
            # new, the gate's predicate is never false, and the pl.when is pure
            # branch overhead -- a known 3.1% left on the table here in
            # exchange for dropping the knob.
            **{**_V3_GENERAL, "kv_slack_pad_lanes": 0},
        ),
    ),
    "decode_f8": dict(
        seq_lens=[(1, 8192)] * 128,
        page_size=256,
        q_dtype=_FP8,
        kv_dtype=_FP8,
        provenance="TUNED_DONATED",
        # Device: v2 0.4798 total / 0.4162 kernel against v3 0.4720 / 0.4114.
        # v3 is 1.6% faster end to end and 1.2% on the kernel -- one of the
        # three specs v3 wins.
        #
        # What flips it is `num_kv_pages_per_block=32`, i.e. bkv_sz = 8192 =
        # the whole sequence, so each sequence is a single k-block and the
        # schedule emits 128 tasks rather than 512. At kv=8 v3 measured 0.5015
        # and lost. The trade is a slightly worse main kernel for a much
        # cheaper schedule pass, and it pays whenever page_size is small enough
        # that the schedule is expensive: it wins here and on
        # decode_f8_kv9216 (both page 256) and is a wash on
        # decode_f8_kv9216_p1024 (page 1024, so 4x fewer pages).
        v2=dict(
            num_kv_pages_per_block=8, num_queries_per_block=1,
            decode_batch_size=8, mixed_q_split=1, vmem_limit_bytes=64 * _MB,
        ),
        v3=dict(
            num_kv_pages_per_block=32, num_queries_per_block=1,
            batch_size=2, n_buffer=2,
            vmem_limit_bytes=64 * _MB, **_V3_GENERAL,
        ),
    ),
    # ---- the three FP8 comparison specs -----------------------------------
    "chunked_prefill_f8_kv1024": dict(
        seq_lens=[(256, 1024)],
        page_size=256,
        q_dtype=_FP8,
        kv_dtype=_FP8,
        provenance="TUNED_DONATED",
        # Swept to 14 v2 configs (top four within 0.6%, so properly tuned) and
        # 12 v3. Device time:
        #
        #                          total    kernel   non-kernel
        #   v2                    0.1719   0.0945      0.0774
        #   v3 (q_split=8)        0.1366   0.0989      0.0377
        #
        # v3 is 1.26x faster end to end while being 1.12x *slower* on the
        # attention kernel -- it wins purely on lower non-kernel overhead,
        # including its own schedule-generation pass. Same pattern as decode.
        #
        # Before `q_split` v3 lost here (0.1868 total, 1.08x slower), because
        # causal block skipping has nothing to skip once history exceeds one
        # bkv_sz: with kv_len 1024 against q_len 256, every q-block needs every
        # k-block. `q_split` is what flips it.
        v2=dict(
            num_kv_pages_per_block=4, num_queries_per_block=32,
            decode_batch_size=1, mixed_q_split=4, vmem_limit_bytes=64 * _MB,
        ),
        v3=dict(
            num_kv_pages_per_block=4, num_queries_per_block=32,
            batch_size=1, n_buffer=2, q_split=8,
            p_same_dtype_as_v=True,  # -4.9% (0.1436 -> 0.1366)
            vmem_limit_bytes=64 * _MB, **_V3_GENERAL,
        ),
    ),
    "chunked_prefill_f8_kv8192": dict(
        seq_lens=[(256, 8192)],
        page_size=256,
        q_dtype=_FP8,
        kv_dtype=_FP8,
        provenance="TUNED_DONATED",
        # Device: v2 0.6104 total / 0.5290 kernel against v3 0.6094 / 0.5703 --
        # v3 is marginally ahead end to end (1.002x) while still 1.08x slower
        # on the kernel, and it gets there computing *more* accurately than v2,
        # which takes the bf16 score approximation by default.
        #
        # Both come from the same mechanism. `schedule.q_loop` emits tasks only
        # for the k-blocks a q-block can causally reach, which halves the work
        # when q_len == kv_len. Here 7936 of 8192 KV tokens are already cached
        # and visible to every query token, so every q-block needs every
        # k-block and there is nothing to skip -- leaving v3's per-step
        # overhead uncompensated.
        #
        # Note the flag polarity is the *decode* one, not prefill's: with most
        # blocks carrying no new KV, `gate_stitch` has real work to elide --
        # turning it off costs 4.1% here (0.7566 vs 0.7265), the opposite sign
        # to prefill_bf16. 27 v3 configs swept.
        #
        # `q_split=16` is what makes `kv=8` reachable at all. bkv_sz=2048 OOMs
        # without it, because the live score tile is [bq_sz * heads, bkv_sz]
        # and q_split divides the first factor; with it, kv=8 beats kv=4 by
        # 3.4% (0.7265 vs 0.7513) since 8192 / 2048 is 4 k-blocks instead of 8.
        # kv=16 and above still OOM at every q_split tried.
        #
        # kv=10 is much worse (0.8957) despite sitting between: 2560 does not
        # divide 8192, so four blocks cover 10240 and the last is mostly waste,
        # where kv=8 tiles the sequence exactly.
        v2=dict(
            num_kv_pages_per_block=8, num_queries_per_block=32,
            decode_batch_size=1, mixed_q_split=4, vmem_limit_bytes=64 * _MB,
        ),
        v3=dict(
            num_kv_pages_per_block=8, num_queries_per_block=32,
            batch_size=1, n_buffer=2, q_split=16,
            # Worth -16.1% here (0.7262 -> 0.6094) and the single change that
            # closes this spec. v2 sets this by default, so until now the
            # comparison was v2 with an fp8 PV operand against v3 with an f32
            # one -- not the same arithmetic. Exactly clean on every shape
            # tested, unlike narrow_scores/narrow_softmax which are 19% wrong
            # on bf16. Prefill-specific: PV has to be large relative to the
            # cast, so it costs 9.4% on decode_f8_kv9216_p1024.
            p_same_dtype_as_v=True,
            vmem_limit_bytes=64 * _MB, **_V3_GENERAL,
        ),
    ),
    "decode_f8_kv9216": dict(
        seq_lens=[(1, 9216)] * 128,
        page_size=256,
        q_dtype=_FP8,
        kv_dtype=_FP8,
        provenance="TUNED_DONATED",
        # The page-256 twin of decode_f8_kv9216_p1024. Device: v2 0.4899 total
        # / 0.4266 kernel against v3 0.5175 / 0.4540 -- v3 1.056x slower, where
        # the page-1024 version is 1.05x. 14 v3 configs swept.
        #
        # Both sides want bkv_sz = 3072-ish for the *kernel*, but v3 does
        # better with kv=36 (bkv 9216, the whole sequence, one k-block per
        # sequence): 0.5175 against 0.5486 at kv=12. Same schedule-amortization
        # trade as decode_f8. kv=18 is worse (0.5588) and batch_size >= 3
        # exceeds VMEM at this block size.
        v2=dict(
            num_kv_pages_per_block=12, num_queries_per_block=1,
            decode_batch_size=8, mixed_q_split=1, vmem_limit_bytes=64 * _MB,
        ),
        v3=dict(
            num_kv_pages_per_block=36, num_queries_per_block=1,
            batch_size=2, n_buffer=2,
            vmem_limit_bytes=64 * _MB, **_V3_GENERAL,
        ),
    ),
    "decode_f8_kv9216_p1024": dict(
        seq_lens=[(1, 9216)] * 128,
        page_size=1024,
        q_dtype=_FP8,
        kv_dtype=_FP8,
        provenance="TUNED_DONATED",
        # Device time at this configuration, after the `_V3_GENERAL` work and
        # with `ql_nope` donated: v2 0.4038 kernel / 0.4665 total; v3 0.4280 /
        # 0.4896 -- 1.060x kernel, 1.050x end to end. Was 1.35x / 1.44x before
        # any of this work.
        #
        # The residual is structural rather than tunable: v3's non-kernel
        # overhead is now *better* than v2's (0.0335 vs 0.0626), and the whole
        # remaining gap is smaller than v3's schedule-generation kernel
        # (0.0428), which v2 has no equivalent of and which cannot overlap
        # because the main kernel consumes its output.
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

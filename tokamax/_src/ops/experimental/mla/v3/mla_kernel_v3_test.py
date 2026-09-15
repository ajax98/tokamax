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
"""Tests for Multi-Head Latent Attention (MLA) V3 kernel."""

import functools
import gc
import sys
import time
from absl import flags
from absl import logging
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.mla import test_base
from tokamax._src.ops.experimental.mla.v2 import kernel as kernel_v2
from tokamax._src.ops.experimental.mla.v3 import mla_wrapper
from tokamax._src.ops.experimental.mla.v3 import utils

FLAGS = flags.FLAGS
flags.DEFINE_bool("debug_mode", False, "Run in debug mode.")

jax.config.parse_flags_with_absl()
jax.config.update("jax_numpy_dtype_promotion", "standard")


class MlaRaggedPagedAttentionKernelV3Test(
    test_base.MlaRaggedPagedAttentionTestBase
):

  def _test_mla_ragged_paged_attention(
      self,
      seq_lens,
      num_heads,
      lkv_dim,
      r_dim,
      page_size,
      q_dtype,
      kv_dtype,
      num_pages,
      *,
      num_kv_pages_per_block=8,
      num_queries_per_block=8,
      vmem_limit_bytes=100 * 1024 * 1024,
      sm_scale=1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      distribution_override: tuple[int, int, int] | None = None,
  ):
    if not jax.devices() or jax.devices()[0].platform != "tpu":
      self.skipTest("Expect TPU")
    if page_size % 128 != 0:
      self.skipTest(
          f"V3 kernel requires page_size to be a multiple of 128 (got {page_size})."
      )
    rng = np.random.default_rng(1234)

    (
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ) = test_base.generate_mla_inputs(
        seq_lens,
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        q_dtype,
        kv_dtype,
        num_pages,
        rng=rng,
    )

    if distribution_override is not None:
      # `generate_mla_inputs` always emits `[d, d, n]`, i.e. an empty
      # pure-prefill band. The reference only reads `distribution[-1]`, so
      # re-banding the same sequences leaves the expected output unchanged and
      # isolates whether the kernel honours the band it is given.
      distribution = jnp.array(distribution_override, dtype=jnp.int32)

    padded_r_dim = test_base.align_to(r_dim, 128)
    padded_lkv_dim = test_base.align_to(lkv_dim, 128)
    padded_kv_dim = padded_lkv_dim + padded_r_dim
    packing = test_base.get_dtype_packing(kv_dtype)
    total_q_len = sum(s[0] for s in seq_lens)
    kv_lens_list = [s[1] for s in seq_lens]
    max_kv_len = max(kv_lens_list) if kv_lens_list else 0
    total_num_pages = max(
        num_pages,
        sum(test_base.cdiv(kv_len, page_size) for kv_len in kv_lens_list),
    )

    expected_out, expected_updated_kv = (
        kernel_v2.ref_mla_ragged_paged_attention(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv.copy(),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    )

    v3_cache_kv = utils.transpose_kv_cache_to_v3(cache_kv.copy(), packing)

    kernel_out, kernel_updated_kv = mla_wrapper.mla_ragged_paged_attention(
        jnp.transpose(ql_nope, (1, 0, 2)),
        q_pe,
        new_kv_c,
        new_k_pe,
        v3_cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=FLAGS.debug_mode,
    )
    kernel_out = jnp.transpose(kernel_out, (1, 0, 2))
    kernel_updated_kv_untransposed = utils.transpose_kv_cache_from_v3(
        kernel_updated_kv, packing
    )
    with np.printoptions(threshold=sys.maxsize):
      logging.vlog(2, "new_kv_c: %s", new_kv_c)
      logging.vlog(2, "new_k_pe: %s", new_k_pe)
      logging.vlog(
          2, "expected_updated_kv.shape: %s", expected_updated_kv.shape
      )
      logging.vlog(
          2, "expected_updated_kv[..., 0]: %s", expected_updated_kv[..., 0]
      )
      logging.vlog(2, "kernel_updated_kv.shape: %s", kernel_updated_kv.shape)
      logging.vlog(
          2, "kernel_updated_kv[..., 0]: %s", kernel_updated_kv[..., 0]
      )

    self.assertEqual(
        expected_out.shape, (total_q_len, num_heads, padded_lkv_dim)
    )
    self.assertEqual(
        expected_updated_kv.shape,
        (total_num_pages, page_size // packing, packing, padded_kv_dim),
    )
    self.assertEqual(
        kernel_updated_kv.shape,
        (total_num_pages, padded_kv_dim // packing, packing, page_size),
    )
    self.assertEqual(
        kernel_updated_kv_untransposed.shape,
        expected_updated_kv.shape,
    )
    self.assertEqual(expected_out.dtype, q_dtype)
    self.assertEqual(expected_updated_kv.dtype, kv_dtype)
    self.assertEqual(kernel_updated_kv.dtype, kv_dtype)

    mask = np.zeros_like(expected_updated_kv, dtype=np.bool_)
    pages_per_seq = test_base.cdiv(max_kv_len, page_size)
    for i, kv_len in enumerate(kv_lens_list):
      start_page_idx_in_pages_list = i * pages_per_seq
      num_pages_for_seq = test_base.cdiv(kv_len, page_size)
      for j in range(num_pages_for_seq):
        page_idx = page_indices[start_page_idx_in_pages_list + j]
        if page_idx == -1:
          logging.warning(
              "Sequence %d page %d has invalid page index -1.", i, j
          )
          continue

        is_last_page = j == num_pages_for_seq - 1
        tokens_on_this_page = (
            kv_len % page_size
            if is_last_page and kv_len % page_size != 0
            else page_size
        )

        for token_idx_in_page in range(tokens_on_this_page):
          row = token_idx_in_page // packing
          col = token_idx_in_page % packing
          mask[page_idx, row, col, :] = True
    true_count = np.sum(mask)
    self.assertEqual(true_count, sum(kv_lens_list) * padded_kv_dim)
    expected_valid = np.array(expected_updated_kv)[mask]
    kernel_valid = np.array(kernel_updated_kv_untransposed)[mask]
    np.testing.assert_array_equal(
        expected_valid,
        kernel_valid,
        err_msg="Updated KV cache mismatch",
    )

    np.testing.assert_allclose(expected_out, kernel_out, atol=0.1, rtol=0.2)
    gc.collect()

  def test_ragged_paged_attention_unaligned_num_q_heads(
      self, dtype=jnp.bfloat16
  ):
    """Head count not divisible by `packing_q`, so `H_pad != H`.

    Every other case uses 128 heads, which every `packing_q` (1/2/4) divides, so
    the padded and unpadded head counts coincide and the causal mask's
    row -> token division cannot be caught getting the divisor wrong.
    """
    seq_lens = [
        (1, 129),
        (1, 122),
        (5, 18),
        (32, 322),
        (3, 1229),
    ]
    self._test_mla_ragged_paged_attention(
        seq_lens,
        127,  # num_heads; bf16 packing_q = 2, so aligned_num_q_heads = 128.
        512,  # lkv_dim
        64,  # r_dim
        128,  # page_size
        dtype,
        self.kv_dtype,
        1024,  # num_pages
    )

  def test_ragged_paged_attention_nonempty_prefill_band(
      self, dtype=jnp.bfloat16
  ):
    """`distribution[1] > distribution[0]`, i.e. a real pure-prefill band.

    Sequences 2 and 3 are routed through the PREFILL pass rather than MIXED.
    Note that their `kv_len > q_len`, so this also covers chunked prefill, where
    the earlier chunks live in the paged cache and must still be attended to.
    """
    seq_lens = [
        (1, 129),
        (1, 122),
        (32, 322),
        (120, 597),
        (5, 18),
        (3, 1229),
    ]
    self._test_mla_ragged_paged_attention(
        seq_lens,
        128,  # num_heads
        512,  # lkv_dim
        64,  # r_dim
        128,  # page_size
        dtype,
        self.kv_dtype,
        1024,  # num_pages
        distribution_override=(2, 4, len(seq_lens)),
    )


class MlaV3DecodePerfTest(absltest.TestCase):
  """Reproduces the best measured v3 decode configuration end to end.

  Workload is the production DeepSeek-V3 decode shape: 128 sequences, one new
  token each, 9216 tokens of context, FP8 throughout, `page_size=1024`. This is
  `decode_f8_kv9216` in `mla/arg_specs.py`.

  Measured on TPU7x (jax 0.11.1), device time from xprof:

      v2  (kv=3, q=1, decode_batch_size=8)     0.4033 ms kernel / 0.4612 total
      v3  this configuration                   0.5445 ms kernel / 0.6628 total

  i.e. v3 is 1.35x v2 on the kernel and 1.44x end to end, down from 1.62x /
  1.68x before the flags below were added.

  Each was measured individually (kernel time, device):

      single-token mask path  -3.4%
      disable_bounds_checks   -6.3%   v2 already compiles with these off
      merged KV DMA           -3.0%   one DMA per cache page instead of two
      kv_slack_pad_lanes=128  -10.5%  makes the stitch stride an odd multiple
                                      of 128; a power-of-two stride aliases
                                      onto VMEM banks and cost +49%

  All but `kv_slack_pad_lanes` are now unconditional kernel behaviour rather
  than flags, so only that one still appears in the config below.

  Deliberately *not* enabled -- all measured counterproductive on decode:
  `narrow_scores` (+1.9%), `p_same_dtype_as_v` (+5.9%), `tight_kv_slack`
  (+49%). The two-step PV deferral is inert here (`q_split == 1`).

  This is a benchmark, not a correctness test; numerics are covered by
  `MlaRaggedPagedAttentionKernelV3Test` and by `mla/v2_v3_op_test.py`. The
  assertion is only a loose regression guard.
  """

  # Generous: the measured wallclock is ~0.93 ms and this is meant to catch a
  # structural regression (a lost flag, a bad config), not to police noise.
  _WALLCLOCK_BUDGET_MS = 2.0

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()
    gc.collect()

  def test_decode_f8_best_config(self):
    if not jax.devices() or jax.devices()[0].platform != "tpu":
      self.skipTest("Expect TPU")

    page_size = 1024
    kv_dtype = q_dtype = jnp.float8_e4m3fn
    seq_lens = [(1, 9216)] * 128

    (
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ) = test_base.generate_mla_inputs(
        seq_lens,
        128,  # num_heads
        512,  # lkv_dim
        64,  # r_dim
        page_size,
        q_dtype,
        kv_dtype,
        128,  # num_pages (a floor; the real count is 128 * 9 = 1152)
        rng=np.random.default_rng(1234),
    )

    # The kernel wants the cache in v3's SEQ_ALONG_LANE layout. Converting here,
    # outside the timed region, is the point: a deployed system holds the cache
    # in this layout permanently, and converting per call costs ~10 ms on this
    # 755 MB cache -- an order of magnitude more than the attention itself.
    packing = test_base.get_dtype_packing(kv_dtype)
    cache_kv = utils.transpose_kv_cache_to_v3(cache_kv, packing)

    ql_nope = jnp.transpose(ql_nope, (1, 0, 2))  # kernel wants head-major

    # `cache_kv` is donated and the updated cache chained into the next call.
    # Without this XLA copies the whole 755 MB cache every iteration (the
    # kernel's `input_output_aliases` wants to write it in place while the
    # caller still owns it), and whether that copy overlaps the kernel or
    # serializes swings the reported time by 2x. Chaining is also what a
    # serving loop does.
    @functools.partial(jax.jit, donate_argnames=("cache_kv",))
    def run(cache_kv):
      return mla_wrapper.mla_ragged_paged_attention(
          ql_nope,
          q_pe,
          new_kv_c,
          new_k_pe,
          cache_kv,
          kv_lens,
          page_indices,
          cu_q_lens,
          distribution,
          num_kv_pages_per_block=3,  # bkv_sz = 3 * 1024; 1,2,4 all measured worse
          num_queries_per_block=1,  # inert on decode (bq_sz is pinned to 1)
          batch_size=4,  # 8 OOMs at kv=3; 1 and 2 are slower
          n_buffer=2,  # 3 is within noise
          kv_slack_pad_lanes=128,
      )

    out, cache_kv = run(cache_kv)
    jax.block_until_ready((out, cache_kv))

    times_ms = []
    for _ in range(10):
      t0 = time.perf_counter()
      out, cache_kv = run(cache_kv)
      jax.block_until_ready((out, cache_kv))
      times_ms.append((time.perf_counter() - t0) * 1e3)

    median_ms = float(np.median(times_ms))
    logging.info(
        "v3 decode_f8 best config: median %.4f ms, min %.4f ms (%d iters,"
        " wallclock)",
        median_ms,
        min(times_ms),
        len(times_ms),
    )
    print(f"\nv3 decode_f8 (page_size=1024) median wallclock: {median_ms:.4f} ms")

    self.assertLess(median_ms, self._WALLCLOCK_BUDGET_MS)


if __name__ == "__main__":
  absltest.main()

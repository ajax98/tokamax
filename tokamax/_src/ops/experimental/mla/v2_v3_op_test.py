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
"""Correctness tests for the MLA v2 and v3 tokamax `Op` wrappers.

Phase 0 exit criteria for the v2-vs-v3 benchmark work: confirm that reaching the
kernels *through the `Op` wrappers* still matches
`v2.kernel.ref_mla_ragged_paged_attention`. The wrappers add head-major/
token-major transposes on both ends, plus a cache-layout conversion for v3, and
those adapters are the likely source of bugs -- the underlying kernels are
already covered by `v2/mla_kernel_v2_test.py` and `v3/mla_kernel_v3_test.py`.

`TransposeRoundTripTest` is pure layout manipulation and runs anywhere JAX is
available. Everything else needs a TPU.
"""

import gc

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.mla import api as mla_api
from tokamax._src.ops.experimental.mla import test_base
from tokamax._src.ops.experimental.mla.v2 import kernel as kernel_v2
from tokamax._src.ops.experimental.mla.v3 import utils as utils_v3

jax.config.update("jax_numpy_dtype_promotion", "standard")


# Matches the FP8 serving configuration the new benchmark arg-specs target.
_FP8 = jnp.float8_e4m3fn


def _skip_unless_tpu(test):
  if not jax.devices() or jax.devices()[0].platform != "tpu":
    test.skipTest("Expect TPU")


# Two dtype settings: `bf16_q` mirrors what `test_base` already validates (bf16
# queries, FP8 cache); `fp8_q` is the serving configuration the benchmark
# arg-specs target.
#
# Both get the same small outlier budget, because neither is exact against the
# reference and for the same underlying reason: the KV cache is FP8 in both
# cases, and v2 further downcasts scores to `s_dtype` (bf16 by default) and `P`
# to the KV dtype. `e4m3` has 3 mantissa bits, so one ULP near 0.5 is 0.0625 --
# a different accumulation order lands a handful of elements a bucket or three
# apart. Measured: v2 misses on 12/16.7M elements (7e-7) at worst; v3, which
# computes in f32 throughout and has no equivalent downcast, matches exactly.
#
# The budget is deliberately 4+ orders of magnitude below what a real defect
# looks like. A head-major/token-major transpose slip misplaces whole rows and
# mismatches ~100% of elements, so this bound still catches the class of bug
# these wrappers can actually introduce.
_MAX_MISMATCH_FRAC = 1e-4

_DTYPE_CASES = {
    "bf16_q": dict(
        q_dtype=jnp.bfloat16,
        kv_dtype=_FP8,
        max_mismatch_frac=_MAX_MISMATCH_FRAC,
    ),
    "fp8_q": dict(
        q_dtype=_FP8, kv_dtype=_FP8, max_mismatch_frac=_MAX_MISMATCH_FRAC
    ),
}


def _assert_close(expected, actual, *, atol, rtol, max_mismatch_frac, msg):
  """`assert_allclose`, but tolerating a bounded fraction of outliers."""
  exp = np.asarray(expected, np.float32)
  act = np.asarray(actual, np.float32)
  bad = ~np.isclose(act, exp, atol=atol, rtol=rtol)
  frac = float(bad.mean())
  if frac > max_mismatch_frac:
    worst = float(np.max(np.abs(act - exp))) if act.size else 0.0
    raise AssertionError(
        f"{msg}: {bad.sum()}/{bad.size} elements ({frac:.3%}) outside "
        f"atol={atol} rtol={rtol}, budget {max_mismatch_frac:.3%}. "
        f"Max absolute difference {worst}."
    )


class TransposeRoundTripTest(parameterized.TestCase):
  """v3 cache-layout conversion is its own inverse. No TPU required."""

  @parameterized.product(
      dtype=(jnp.bfloat16, _FP8),
      page_size=(128, 256),
  )
  def test_cache_round_trip(self, dtype, page_size):
    packing = test_base.get_dtype_packing(dtype)
    pages, kv_dim = 8, 640
    rng = np.random.default_rng(0)
    cache = jnp.asarray(
        rng.integers(0, 8, size=(pages, page_size // packing, packing, kv_dim)),
        dtype=dtype,
    )

    v3_cache = utils_v3.transpose_kv_cache_to_v3(cache, packing)
    self.assertEqual(
        v3_cache.shape, (pages, kv_dim // packing, packing, page_size)
    )

    round_tripped = utils_v3.transpose_kv_cache_from_v3(v3_cache, packing)
    self.assertEqual(round_tripped.shape, cache.shape)
    np.testing.assert_array_equal(np.asarray(cache), np.asarray(round_tripped))


class MlaOpWrapperTest(parameterized.TestCase):
  """Both `Op` wrappers must agree with the v2 reference."""

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()
    gc.collect()

  def _run(
      self, impl_name, seq_lens, *, dtype_case="bf16_q", page_size=256,
      num_heads=128,
  ):
    _skip_unless_tpu(self)

    case = _DTYPE_CASES[dtype_case]
    q_dtype, kv_dtype = case["q_dtype"], case["kv_dtype"]
    lkv_dim, r_dim, num_pages = 512, 64, 1024
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
        rng=np.random.default_rng(1234),
    )

    expected_out, expected_cache = kernel_v2.ref_mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )

    impl = mla_api.IMPLEMENTATIONS[impl_name]

    # Wrapped in a lambda rather than `jax.jit(impl)` so the `Op` instance stays
    # a Python closure constant instead of a traced argument. Matches how
    # `benchmarks/mla.py` jits through `mla_op_wrapper`.
    @jax.jit
    def run(ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv):
      return impl(
          ql_nope=ql_nope,
          q_pe=q_pe,
          new_kv_c=new_kv_c,
          new_k_pe=new_k_pe,
          cache_kv=cache_kv,
          kv_lens=kv_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
      )

    actual_out, actual_cache = run(
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv.copy()
    )

    # Shape is the point of the test: the wrappers transpose on both ends, so a
    # head-major/token-major slip would show up here rather than as a numeric
    # mismatch (the two are the same shape only when T == H).
    self.assertEqual(expected_out.shape, actual_out.shape)
    self.assertEqual(expected_cache.shape, actual_cache.shape)
    self.assertEqual(expected_out.dtype, actual_out.dtype)

    _assert_close(
        expected_out,
        actual_out,
        atol=0.1,
        rtol=0.2,
        max_mismatch_frac=case["max_mismatch_frac"],
        msg=f"{impl_name}/{dtype_case} output mismatch",
    )

  @parameterized.product(
      impl_name=("v2", "v3"), dtype_case=tuple(_DTYPE_CASES)
  )
  def test_decode(self, impl_name, dtype_case):
    """Pure decode band; `distribution == [n, n, n]`."""
    self._run(impl_name, [(1, 1024), (1, 768), (1, 2048)], dtype_case=dtype_case)

  @parameterized.product(
      impl_name=("v2", "v3"), dtype_case=tuple(_DTYPE_CASES)
  )
  def test_chunked_prefill(self, impl_name, dtype_case):
    """`kv_len > q_len`, so history lives in the paged cache."""
    self._run(impl_name, [(256, 1024)], dtype_case=dtype_case)

  @parameterized.product(
      impl_name=("v2", "v3"), dtype_case=tuple(_DTYPE_CASES)
  )
  def test_ragged(self, impl_name, dtype_case):
    """Skewed lengths -- where v2 pads to the batch max and v3 does not."""
    self._run(
        impl_name,
        [(1, 128), (1, 4096), (1, 512), (1, 2048)],
        dtype_case=dtype_case,
    )

  @parameterized.product(
      impl_name=("v2", "v3"), dtype_case=tuple(_DTYPE_CASES)
  )
  def test_non_square_token_head_counts(self, impl_name, dtype_case):
    """`total_q_len != num_heads`, so a transpose slip cannot pass silently."""
    self._run(
        impl_name, [(1, 512), (1, 512)], num_heads=128, dtype_case=dtype_case
    )

  def test_both_registered(self):
    """Guards the `api.py` registration itself, which needs no TPU."""
    self.assertIn("v2", mla_api.IMPLEMENTATIONS)
    self.assertIn("v3", mla_api.IMPLEMENTATIONS)


class V3PrecisionKnobTest(parameterized.TestCase):
  """v3's ported v2 inner-loop knobs must not break numerics.

  These narrow intermediates (scores to bf16, softmax probabilities to the KV
  dtype), so some accuracy loss is expected and intended -- that is the trade
  being made for MXU/VREG throughput. The bar is that they stay within the same
  outlier budget the unmodified kernels already need against the reference, not
  that they are bit-identical.
  """

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()
    gc.collect()

  @parameterized.named_parameters(
      ("baseline", {}, None, None),
      ("narrow_scores", dict(narrow_scores=True), None, None),
      ("p_same_dtype_as_v", dict(p_same_dtype_as_v=True), None, None),
      ("both", dict(narrow_scores=True, p_same_dtype_as_v=True), None, None),
      # `fast_mask` rewrites the causal predicate for single-token query
      # blocks, so it needs its own coverage -- and a sliding-window case,
      # since that inequality is rearranged too
      # (`q_kv_diff < sw + offset` becomes `kv_iota > -offset - sw`).
      ("fast_mask", dict(fast_mask=True), None, None),
      ("fast_mask_sliding_window", dict(fast_mask=True), 512, None),
      ("general_mask_sliding_window", {}, 512, None),
      # Prefill shape, both ways. The control is essential: with
      # `num_queries_per_block=1` the prefill band also has bq_sz==1, so the
      # fast path applies there too -- and without a fast_mask=False run on the
      # same shape there is no way to tell a fast-path bug from a pre-existing
      # one in v3's bq_sz==1 prefill handling.
      ("fast_mask_prefill", dict(fast_mask=True), None, [(256, 1024)]),
      ("general_mask_prefill", {}, None, [(256, 1024)]),
      # And with bq_sz > 1, where the fast path must not engage at all.
      ("fast_mask_prefill_bq16", dict(fast_mask=True), None, [(256, 1024)]),
      (
          "all_knobs",
          dict(narrow_scores=True, p_same_dtype_as_v=True, fast_mask=True),
          None,
          None,
      ),
      # DMA / VMEM knobs. These are pure plumbing changes -- one page of stitch
      # slack instead of two, one DMA per page instead of two, bounds checks
      # off -- so unlike the precision knobs they should be *numerically
      # identical*, not merely close. Tested against the reference anyway, and
      # combined, since tight_kv_slack shrinks the very buffer merge_kv_dma
      # writes into.
      ("tight_kv_slack", dict(tight_kv_slack=True), None, None),
      ("merge_kv_dma", dict(merge_kv_dma=True), None, None),
      ("no_bounds_checks", dict(disable_bounds_checks=True), None, None),
      (
          "dma_knobs_combined",
          dict(tight_kv_slack=True, merge_kv_dma=True,
               disable_bounds_checks=True, fast_mask=True),
          None,
          None,
      ),
      # tight_kv_slack must NOT engage for a multi-token query block; this
      # exercises the bq_sz>1 path with the flag on.
      ("tight_kv_slack_prefill_bq16", dict(tight_kv_slack=True), None,
       [(256, 1024)]),
  )
  def test_knob_matches_reference(self, knobs, sliding_window, seq_lens):
    _skip_unless_tpu(self)
    from tokamax._src.ops.experimental.mla import v3_op  # pylint: disable=g-import-not-at-top

    if seq_lens is None:
      seq_lens = [(1, 1024), (1, 768), (1, 2048)]
    (
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
        kv_lens, page_indices, cu_q_lens, distribution,
    ) = test_base.generate_mla_inputs(
        seq_lens, 128, 512, 64, 256, _FP8, _FP8, 1024,
        rng=np.random.default_rng(1234),
    )

    expected_out, _ = kernel_v2.ref_mla_ragged_paged_attention(
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv.copy(),
        kv_lens, page_indices, cu_q_lens, distribution,
        sliding_window=sliding_window,
    )

    # bq_sz>1 only for the explicitly-named variant; everything else keeps
    # num_queries_per_block=1 so the decode-shaped path is what is exercised.
    nqpb = 16 if "bq16" in self._testMethodName else 1
    config = v3_op.Config(
        num_kv_pages_per_block=8,
        num_queries_per_block=nqpb,
        batch_size=2,
        n_buffer=2,
        vmem_limit_bytes=64 * 1024 * 1024,
        **knobs,
    )
    op = mla_api.IMPLEMENTATIONS["v3"].replace(config=config)

    @jax.jit
    def run(cache_kv):
      return op(
          ql_nope=ql_nope, q_pe=q_pe, new_kv_c=new_kv_c, new_k_pe=new_k_pe,
          cache_kv=cache_kv, kv_lens=kv_lens, page_indices=page_indices,
          cu_q_lens=cu_q_lens, distribution=distribution,
          sliding_window=sliding_window,
      )

    actual_out, _ = run(cache_kv.copy())

    self.assertEqual(expected_out.shape, actual_out.shape)
    _assert_close(
        expected_out,
        actual_out,
        atol=0.1,
        rtol=0.2,
        max_mismatch_frac=_MAX_MISMATCH_FRAC,
        msg=f"v3 with {knobs or 'defaults'} sw={sliding_window} mismatch",
    )


class V3BlockOverrunTest(parameterized.TestCase):
  """Isolates a pre-existing v3 mismatch when `bkv_sz` exceeds `kv_len`.

  Found while validating `fast_mask`: a prefill shape with
  `num_kv_pages_per_block=8` (bkv_sz = 2048) against `kv_len = 1024` mismatches
  the reference on 1.37% of elements, and does so identically with the mask
  fast path disabled -- so it is not the fast path. `test_chunked_prefill`
  passes on the same shape because the heuristics config uses
  `num_kv_pages_per_block=4` (bkv_sz = 1024), exactly the sequence length.

  This sweeps `num_kv_pages_per_block` at fixed everything-else to find where
  it breaks. Decode is unaffected -- `kv_len = 9216` there, far above any
  `bkv_sz` in the search space -- but the autotuner does explore this region,
  so a config that is fast *and wrong* could be selected on prefill shapes.
  """

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()
    gc.collect()

  @parameterized.named_parameters(
      ("kv1_bkv256", 1),
      ("kv2_bkv512", 2),
      ("kv4_bkv1024", 4),
      ("kv8_bkv2048", 8),
      ("kv16_bkv4096", 16),
  )
  def test_kv_block_vs_seq_len(self, num_kv_pages_per_block):
    _skip_unless_tpu(self)
    from tokamax._src.ops.experimental.mla import v3_op  # pylint: disable=g-import-not-at-top

    seq_lens = [(256, 1024)]  # kv_len = 1024; bkv_sz = n * 256
    (
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
        kv_lens, page_indices, cu_q_lens, distribution,
    ) = test_base.generate_mla_inputs(
        seq_lens, 128, 512, 64, 256, _FP8, _FP8, 1024,
        rng=np.random.default_rng(1234),
    )

    expected_out, _ = kernel_v2.ref_mla_ragged_paged_attention(
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv.copy(),
        kv_lens, page_indices, cu_q_lens, distribution,
    )

    op = mla_api.IMPLEMENTATIONS["v3"].replace(
        config=v3_op.Config(
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=4,
            batch_size=2,
            n_buffer=2,
            vmem_limit_bytes=64 * 1024 * 1024,
        )
    )

    @jax.jit
    def run(cache_kv):
      return op(
          ql_nope=ql_nope, q_pe=q_pe, new_kv_c=new_kv_c, new_k_pe=new_k_pe,
          cache_kv=cache_kv, kv_lens=kv_lens, page_indices=page_indices,
          cu_q_lens=cu_q_lens, distribution=distribution,
      )

    actual_out, _ = run(cache_kv.copy())
    _assert_close(
        expected_out, actual_out, atol=0.1, rtol=0.2,
        max_mismatch_frac=_MAX_MISMATCH_FRAC,
        msg=f"bkv_sz={num_kv_pages_per_block * 256} vs kv_len=1024",
    )


class V3CompactKvDimTest(parameterized.TestCase):
  """`compact_kv_dim` must be a pure layout change, not a numerics change.

  v3 stores kv_dim on sublanes, so padding each part to 128 lanes -- inherited
  from v2, where kv_dim is on lanes -- wastes 10%: align_to(512,128) +
  align_to(64,128) = 640 against a true sum of 576. The dropped 64 rows are
  padding zeros that contribute nothing to the QK-PE dot, so removing them must
  leave the output bit-comparable.

  Built by slicing the padded cache rather than generating fresh random data,
  so the two layouts hold *identical* values and any difference is the layout
  change itself.
  """

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()
    gc.collect()

  @parameterized.named_parameters(
      ("decode", [(1, 1024), (1, 768), (1, 2048)], False),
      ("decode_merged_dma", [(1, 1024), (1, 768), (1, 2048)], True),
      ("chunked_prefill", [(256, 1024)], False),
  )
  def test_compact_matches_reference(self, seq_lens, merge):
    _skip_unless_tpu(self)
    from tokamax._src.ops.experimental.mla import v3_op  # pylint: disable=g-import-not-at-top

    packing = test_base.get_dtype_packing(_FP8)
    (
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
        kv_lens, page_indices, cu_q_lens, distribution,
    ) = test_base.generate_mla_inputs(
        seq_lens, 128, 512, 64, 256, _FP8, _FP8, 1024,
        rng=np.random.default_rng(1234),
    )

    expected_out, _ = kernel_v2.ref_mla_ragged_paged_attention(
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv.copy(),
        kv_lens, page_indices, cu_q_lens, distribution,
    )

    # Padded v3 layout: [pages, 640/P, P, page_size]. Sublanes 0..127 are the
    # 512-wide latent part; 128..159 are the RoPE part padded to 128, of which
    # only the first 16 sublanes (64 values) carry data.
    padded = utils_v3.transpose_kv_cache_to_v3(cache_kv, packing)
    lkv_sub = 512 // packing            # 128
    r_sub = 64 // packing               # 16
    compact = jnp.concatenate(
        [padded[:, :lkv_sub], padded[:, lkv_sub:lkv_sub + r_sub]], axis=1
    )
    self.assertEqual(compact.shape[1], (512 + 64) // packing)  # 144

    base = dict(
        num_kv_pages_per_block=8,
        num_queries_per_block=4,
        batch_size=2,
        n_buffer=2,
        vmem_limit_bytes=64 * 1024 * 1024,
        merge_kv_dma=merge,
    )
    op_native = mla_api.IMPLEMENTATIONS["v3_native"]

    def run_with(cache, **knobs):
      o = op_native.replace(config=v3_op.Config(**base, **knobs))

      @jax.jit
      def f(c):
        return o(
            ql_nope=ql_nope, q_pe=q_pe, new_kv_c=new_kv_c, new_k_pe=new_k_pe,
            cache_kv=c, kv_lens=kv_lens, page_indices=page_indices,
            cu_q_lens=cu_q_lens, distribution=distribution,
        )

      return f(cache)[0]

    out_padded = run_with(padded)
    out_compact = run_with(compact, compact_kv_dim=True)

    # Both against the reference...
    for label, out in (("padded", out_padded), ("compact", out_compact)):
      _assert_close(
          expected_out, out, atol=0.1, rtol=0.2,
          max_mismatch_frac=_MAX_MISMATCH_FRAC,
          msg=f"v3_native {label} vs reference",
      )
    # ...and against each other, which is the tighter check: the padding is
    # zeros, so dropping it should not perturb the result at all.
    _assert_close(
        out_padded, out_compact, atol=0.0, rtol=0.0,
        max_mismatch_frac=0.0,
        msg="compact vs padded layout differ",
    )


if __name__ == "__main__":
  absltest.main()

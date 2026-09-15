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
"""Tokamax `Op` wrapper for the experimental MLA v3 kernel.

Exists so that `v3/mla_wrapper.py` can be reached by `tokamax.autotune` and by
`tokamax/benchmarks/mla.py`, both of which dispatch through
`mla.api.IMPLEMENTATIONS` and therefore only see registered `Op`s.

Two things differ structurally from `v2_op.py` and are worth reading before
interpreting any benchmark number:

  1. **Cache layout.** v3 uses a transposed 4D cache,
     `[pages, aligned_kv_dim // P, P, page_size]`, which is *not* the layout the
     `Op` contract supplies. This wrapper therefore converts on the way in and
     back on the way out, and **those conversions are inside the timed region.**
     In a real serving stack the cache would simply live in v3 layout, so this
     measures drop-in migration cost rather than steady-state kernel cost. A
     no-conversion variant is deferred; see the plan doc.

  2. **No precision knobs.** v3 has no `s_dtype` and no `p_same_dtype_as_v`
     equivalent -- scores and softmax probabilities stay f32 throughout. The
     base `s_dtype` argument is accepted and ignored.
"""

import itertools
from typing import Annotated, ClassVar, override

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.mla import base
from tokamax._src.ops.experimental.mla.v3 import mla_wrapper
from tokamax._src.ops.experimental.mla.v3 import utils as utils_v3


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Tuning configuration for the MLA v3 kernel.

  Attributes:
    num_kv_pages_per_block: KV pages per block; sets `bkv_sz = n * page_size`.
    num_queries_per_block: Query tokens per block for the PREFILL and MIXED
      passes. The DECODE pass always uses `bq_sz = 1` regardless, which is what
      selects the O(1) stitch path in `stitch_utils`.
    batch_size: Number of *consecutive schedule tasks* fused into one grid step.
      Not the v2 `decode_batch_size`: the lanes share a single `(m, l, acc)`
      chained serially, so this is a work-per-step / DMA-overlap knob rather
      than a parallelism knob.
    n_buffer: Pipeline buffer depth. v2 is fixed at double buffering; v3 can go
      deeper, which is one of the few places it has a knob v2 lacks.
    vmem_limit_bytes: VMEM budget passed to the Mosaic compiler. Note v3's KV
      staging buffer carries `2 * page_size` lanes of stitch slack against v2's
      constant `3 * 128`, so v3's VMEM pressure grows with `page_size`.
  """

  num_kv_pages_per_block: Annotated[int, pydantic.Field(gt=0)]
  num_queries_per_block: Annotated[int, pydantic.Field(gt=0)]
  batch_size: Annotated[int, pydantic.Field(gt=0)]
  n_buffer: Annotated[int, pydantic.Field(ge=2)]
  vmem_limit_bytes: Annotated[int, pydantic.Field(multiple_of=16, gt=0)]
  # Twenty flags accumulated here over the v3 optimization effort; six remain.
  # Twelve were removed as always-bad or always-zero, and eight more (flat_q,
  # fuse_qk, disable_bounds_checks, fast_mask, static_kv_dst, merge_kv_dma,
  # gate_stitch, two_step_flash_attention) became unconditional kernel
  # behaviour once they measured as wins everywhere -- a flag nobody should
  # ever set to False is not a tuning knob. See the optimization log for the
  # per-spec numbers behind each.
  #
  # What is left is only what genuinely depends on the shape.
  #
  #   kv_slack_pad_lanes  128 on decode (-10.5%, breaks a power-of-two VMEM
  #                       stride), 0 on whole-prompt prefill. Only matters when
  #                       the stride lands on a power of two, and the exact
  #                       rule is bkv_sz- and page_size-specific -- the "make
  #                       kv_vmem_lanes // 128 odd" shortcut fits decode but
  #                       does not hold on prefill, so this stays measured.
  #   p_same_dtype_as_v   -16.1% on chunked_prefill_f8_kv8192, +9.4% on
  #                       decode_f8_kv9216_p1024. NOTE the mechanism is not
  #                       established. It is NOT row-count amortization: the
  #                       PV tile is 128 rows on decode and 256 on kv8192, a
  #                       2x difference that cannot explain a 25-point swing.
  #                       HLO keeps dot(f32, fp8) as one dot_general with no
  #                       convert, so whatever reconciles the operand formats
  #                       happens inside the backend and is not visible here.
  #                       kv8192 is the most arithmetic-bound spec in the set
  #                       (no causal skipping -- 7936 of 8192 KV tokens are
  #                       cached and visible to every query), which is the
  #                       likeliest reason it is the one that benefits.
  #                       See bench_pv_operand_dtypes.py to settle it.
  #   q_split             Inert on decode (bq_sz is 1). Worth -9.7% on prefill.
  #                       Cannot be defaulted to a constant: the right value
  #                       trades against VMEM, and too small OOMs
  #                       (chunked_prefill_f8_kv8192 needs 16; 8 does not fit).
  kv_slack_pad_lanes: int = 128
  p_same_dtype_as_v: bool = False
  q_split: int = 1

  # --- Situational.


class V3MultiHeadLatentAttention(base.MultiHeadLatentAttention):
  """Tokamax operator invoking the experimental MLA v3 kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      ql_nope: Float[Array, "max_num_tokens actual_num_q_heads actual_lkv_dim"],
      q_pe: Float[Array, "max_num_tokens actual_num_q_heads actual_r_dim"],
      new_kv_c: Float[Array, "max_num_tokens actual_lkv_dim"],
      new_k_pe: Float[Array, "max_num_tokens actual_r_dim"],
      cache_kv: Float[
          Array,
          "total_num_pages page_size_per_kv_packing kv_packing lkv_dim",
      ],
      kv_lens: Int[Array, "max_num_seqs"],
      page_indices: Int[Array, "num_page_indices"],
      cu_q_lens: Int[Array, "max_num_seqs_plus_1"],
      distribution: Int[Array, "3"],
      *,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      s_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      debug_mode: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    # v3 computes scores and probabilities in f32 unconditionally and exposes no
    # equivalent knob, so `s_dtype` cannot be honoured. Accepted and dropped
    # rather than raising, so the same `ArgSpec` can drive both v2 and v3.
    del s_dtype, return_residuals

    assert config is not None, "Config must be specified."

    # `[pages, page_size // P, P, kv_dim]` -> `[pages, kv_dim // P, P, page_size]`.
    packing = cache_kv.shape[2]
    cache_kv_v3 = utils_v3.transpose_kv_cache_to_v3(cache_kv, packing)

    # The `Op` contract is token-major `[T, H, L]`; v3, like v2, historically
    # took and returned head-major `[H, T, L]`. `q_pe` is token-major in both.
    #
    # That contract cost four full passes over an 8.4 MB array on the decode
    # shape, all cancelling: the transpose here was undone by
    # `prepare_q_nope_inputs`, and `prepare_outputs`' transpose was undone
    out, updated_cache_kv_v3 = mla_wrapper.mla_ragged_paged_attention(
        jnp.transpose(ql_nope, (1, 0, 2)),
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv_v3,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        num_kv_pages_per_block=config.num_kv_pages_per_block,
        num_queries_per_block=config.num_queries_per_block,
        batch_size=config.batch_size,
        n_buffer=config.n_buffer,
        vmem_limit_bytes=config.vmem_limit_bytes,
        s_dtype=None,
        p_same_dtype_as_v=config.p_same_dtype_as_v,
        kv_slack_pad_lanes=config.kv_slack_pad_lanes,
        q_split=config.q_split,
        debug_mode=debug_mode,
    )

    updated_cache_kv = utils_v3.transpose_kv_cache_from_v3(
        updated_cache_kv_v3, packing
    )
    out = jnp.transpose(out, (1, 0, 2))
    return (out, updated_cache_kv), None

  @override
  def _get_heuristics_config(self, ba) -> Config:
    # Must fit VMEM for *every* shape, since this is the untuned fallback.
    #
    # `num_queries_per_block` only affects the PREFILL/MIXED passes (DECODE
    # pins `bq_sz = 1`), and it is the parameter that blows the budget: the f32
    # accumulator is `[bq_sz * aligned_num_q_heads, aligned_lkv_dim]`, so at
    # bq_sz=16 with 128 heads that is 16*128*512*4 = 4.2 MB on its own, before
    # the `batch_size * n_buffer` copies of the Q and O staging buffers. A
    # measured run at (kv=8, q=16, batch=2, n_buffer=2) needed 84.16 MB against
    # a 63.94 MB budget on a chunked-prefill shape.
    #
    # 4 matches `mla_wrapper.calculate_block_sizes`'s own default and leaves
    # room at page_size=1024, where v3's `bkv_sz + 2 * page_size` KV slack is
    # much more expensive than v2's constant `+ 3 * 128`.
    return Config(
        num_kv_pages_per_block=4,
        num_queries_per_block=4,
        batch_size=2,
        n_buffer=2,
        vmem_limit_bytes=64 * 1024 * 1024,
    )

  @override
  def _get_autotuning_configs(self, ba) -> set[Config]:
    configs = set()
    for kv, q, batch_size, n_buffer in itertools.product(
        # 3 mirrors the production DS-V3 decode sweep; it is not a power of two.
        [1, 2, 3, 4, 8, 16],
        # 1 included to match v2's [1, 4, 16]. On a decode-only distribution
        # this axis is inert -- v3 pins bq_sz=1 in `calculate_block_sizes` and
        # v2 clamps via `min(num_queries_per_block, static_q_len=1)` -- but it
        # is live for prefill/mixed, where omitting 1 gave v2 a config v3 could
        # not reach.
        [1, 4, 16],
        [1, 2, 4],
        [2, 3],
    ):
      configs.add(
          Config(
              num_kv_pages_per_block=kv,
              num_queries_per_block=q,
              batch_size=batch_size,
              n_buffer=n_buffer,
              vmem_limit_bytes=64 * 1024 * 1024,
          )
      )
    return configs

  @override
  def supported_on(self, device) -> bool:
    # v3 additionally requires `page_size` to be a power of two and a multiple
    # of 128, and `bkv_sz % page_size == 0`. Those depend on the arguments, not
    # the device, so they stay as runtime checks in
    # `v3.kernel.static_validate_inputs`.
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5

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

"""Wrapper for MLA v3 kernel to orchestrate multi-phase batch execution."""

from absl import logging
import jax
from jax import lax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import kernel
from tokamax._src.ops.experimental.mla.v3 import schedule


# Matches `v2/kernel.py`'s DEFAULT_MASK_VALUE. Deliberately not
# `finfo(float32).min`: the masked logits are fed straight into the online
# softmax, and a value that close to the representable limit turns the
# `s - m` subtraction into -inf (and then 0 * inf -> NaN) for any block whose
# rows are entirely masked. 0.7 of the max leaves room for that subtraction
# while still driving exp() to zero.
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def calculate_block_sizes(
    serve_cfgs: configs.ServingConfigs,
    *,
    num_kv_pages_per_block: int = 2,
    num_queries_per_block: int = 4,
    batch_size: int = 2,
    n_buffer: int = 2,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
  """Calculates default block sizes for decode and prefill/mixed passes."""
  page_size = serve_cfgs.page_size
  bkv_sz = num_kv_pages_per_block * page_size

  decode_blocks = configs.BlockSizes(
      bq_sz=1,
      bq_c_sz=1,
      bkv_sz=bkv_sz,
      batch_size=batch_size,
      n_buffer=n_buffer,
  )
  prefill_blocks = configs.BlockSizes(
      bq_sz=num_queries_per_block,
      bq_c_sz=num_queries_per_block,
      bkv_sz=bkv_sz,
      batch_size=batch_size,
      n_buffer=n_buffer,
  )
  return decode_blocks, prefill_blocks


def mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim] or [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.Array,  # [total_num_pages, aligned_kv_dim // kv_packing, kv_packing, page_size]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int = 2,
    num_queries_per_block: int = 4,
    batch_size: int = 2,
    n_buffer: int = 2,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    s_dtype=None,
    p_same_dtype_as_v: bool = False,
    two_step_flash_attention: bool = False,
    fast_mask: bool = False,
    compact_kv_dim: bool = False,
    tight_kv_slack: bool = False,
    merge_kv_dma: bool = False,
    disable_bounds_checks: bool = False,
    kv_slack_pad_lanes: int = 0,
    debug_mode: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """MLA Ragged paged attention, orchestrated as Decode -> Prefill -> Mixed."""
  actual_num_q_heads, total_q_tokens, actual_lkv_dim = ql_nope.shape

  actual_r_dim = q_pe.shape[-1]
  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]

  if vmem_limit_bytes is None:
    vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE

  page_size = cache_kv.shape[-1]

  model_cfgs = configs.MlaModelConfigs(
      num_q_heads=actual_num_q_heads,
      lkv_dim=actual_lkv_dim,
      r_dim=actual_r_dim,
      mask_value=mask_value,
      sm_scale=sm_scale,
      soft_cap=soft_cap,
      sliding_window=sliding_window,
  )
  serve_cfgs = configs.ServingConfigs(
      num_seqs=max_num_seqs,
      num_page_indices=num_page_indices,
      total_q_tokens=total_q_tokens,
      dtype_q=ql_nope.dtype,
      dtype_kv=cache_kv.dtype,
      dtype_out=ql_nope.dtype,
      page_size=page_size,
      scale_q=q_scale,
      scale_k=k_scale,
      scale_v=v_scale,
      kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
      s_dtype=s_dtype,
      p_same_dtype_as_v=p_same_dtype_as_v,
      two_step_flash_attention=two_step_flash_attention,
      fast_mask=fast_mask,
      compact_kv_dim=compact_kv_dim,
      tight_kv_slack=tight_kv_slack,
      merge_kv_dma=merge_kv_dma,
      disable_bounds_checks=disable_bounds_checks,
      kv_slack_pad_lanes=kv_slack_pad_lanes,
  )

  default_decode, default_prefill = calculate_block_sizes(
      serve_cfgs,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      batch_size=batch_size,
      n_buffer=n_buffer,
  )

  init_cfgs = configs.MlaConfigs(
      block=default_decode,
      model=model_cfgs,
      serve=serve_cfgs,
      vmem_limit_bytes=vmem_limit_bytes,
      mode=configs.MlaCase.DECODE,
  )
  kernel.static_validate_inputs(
      ql_nope,
      q_pe,
      new_kv_c,
      new_k_pe,
      cache_kv,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
      cfgs=init_cfgs,
  )

  ql_nope_prep = kernel.prepare_q_nope_inputs(
      ql_nope,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  # All three stay at 128-lane alignment. `q_pe` in particular *must*: it is
  # reshaped with `aligned_r_dim` as its minor dimension, which is the lane
  # axis. See `MlaConfigs.kv_dim_align`.
  head_align = init_cfgs.kv_dim_align
  q_pe_prep = kernel.prepare_q_inputs(q_pe, head_align=head_align)
  new_kv_c_prep = kernel.prepare_kv_inputs_for_transposed_kv_cache(
      new_kv_c,
      page_size=page_size,
      head_align=head_align,
  )
  # `new_k_pe` is KV, so it follows the *HBM* width and shrinks with
  # `compact_kv_dim`. `q_pe` above must not: it is reshaped with its head
  # dimension as the minor (lane) axis.
  new_k_pe_prep = kernel.prepare_kv_inputs_for_transposed_kv_cache(
      new_k_pe,
      page_size=page_size,
      head_align=(
          init_cfgs.serve.packing_kv * 8 if compact_kv_dim else head_align
      ),
  )

  def run_mla_kernel(
      mode: configs.MlaCase,
      ql_nope_in: jax.Array,
      cache_kv_in: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    effective_blocks = (
        decode_block_sizes or default_decode
        if mode == configs.MlaCase.DECODE
        else prefill_block_sizes or default_prefill
    )
    if debug_mode:
      logging.info("blocks: %s, mode: %s", effective_blocks, mode)
    cfgs = configs.MlaConfigs(
        block=effective_blocks,
        model=model_cfgs,
        serve=serve_cfgs,
        vmem_limit_bytes=vmem_limit_bytes,
        mode=mode,
    )
    schedule_hbm = schedule.generate_mla_metadata(
        cu_q_lens, kv_lens, page_indices, distribution, cfgs=cfgs
    )
    return kernel._mla_ragged_paged_attention_kernel(
        cu_q_lens,
        kv_lens,
        page_indices,
        schedule_hbm,
        ql_nope_in,
        q_pe_prep,
        new_kv_c_prep,
        new_k_pe_prep,
        cache_kv_in,
        cfgs=cfgs,
    )

  num_decode = distribution[0]
  num_prefill = distribution[1] - distribution[0]
  num_mixed = distribution[2] - distribution[1]

  if debug_mode:
    logging.info(
        "Prepared inputs for MLA: ql_nope=%s, q_pe=%s, new_kv_c=%s,"
        " new_k_pe=%s, cache_kv=%s",
        ql_nope_prep.shape,
        q_pe_prep.shape,
        new_kv_c_prep.shape,
        new_k_pe_prep.shape,
        cache_kv.shape,
    )

  # `distribution` partitions the sequences into three contiguous bands:
  #   [0, d0)   decode          -- q_len == 1
  #   [d0, d1)  chunked prefill -- q_len > 1, static
  #   [d1, d2)  mixed
  # Each band gets its own pass, chaining `ql_nope_prep` (which aliases the
  # output buffer) and `cache_kv` so later passes see earlier writes. The
  # prefill band used to be skipped entirely, which silently dropped those
  # sequences from the output whenever d1 > d0.
  def run_pass(mode, carry, predicate):
    return lax.cond(
        predicate,
        lambda q_kv: run_mla_kernel(mode, q_kv[0], q_kv[1]),
        lambda q_kv: q_kv,
        carry,
    )

  carry = (ql_nope_prep, cache_kv)
  carry = run_pass(configs.MlaCase.DECODE, carry, num_decode > 0)
  carry = run_pass(configs.MlaCase.PREFILL, carry, num_prefill > 0)
  o_hbm, cache_kv = run_pass(configs.MlaCase.MIXED, carry, num_mixed > 0)

  output = kernel.prepare_outputs(
      o_hbm,
      actual_num_q_heads,
      total_q_tokens,
      actual_lkv_dim,
      vmem_limit_bytes=vmem_limit_bytes,
  )

  return output, cache_kv

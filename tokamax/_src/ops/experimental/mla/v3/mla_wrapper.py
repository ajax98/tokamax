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

import functools
from absl import logging
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import kernel
from tokamax._src.ops.experimental.mla.v3 import schedule
from tokamax._src.ops.experimental.mla.v3 import utils


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
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """MLA Ragged paged attention with multi-phase (Decode -> Mixed) orchestration."""
  actual_num_q_heads, total_q_tokens, actual_lkv_dim = ql_nope.shape
  is_head_first = True

  actual_r_dim = q_pe.shape[-1]
  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]

  if vmem_limit_bytes is None:
    vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
  if mask_value is None:
    mask_value = float(jnp.finfo(jnp.float32).min)

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
  )

  default_decode, default_prefill = calculate_block_sizes(
      serve_cfgs,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
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
  q_pe_prep = kernel.prepare_q_inputs(q_pe)
  new_kv_c_prep = kernel.prepare_kv_inputs_for_transposed_kv_cache(
      new_kv_c,
      page_size=page_size,
  )
  new_k_pe_prep = kernel.prepare_kv_inputs_for_transposed_kv_cache(
      new_k_pe,
      page_size=page_size,
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
  num_mixed = distribution[2] - distribution[1]

  # Pass 1: Decode (runs sequences 0..distribution[0]-1 only if num_decode > 0)
  ql_nope_prep, cache_kv = lax.cond(
      num_decode > 0,
      lambda q_kv: run_mla_kernel(configs.MlaCase.DECODE, q_kv[0], q_kv[1]),
      lambda q_kv: q_kv,
      (ql_nope_prep, cache_kv),
  )

  # Pass 2: Mixed (runs sequences distribution[1]..total_seqs-1 only if num_mixed > 0)
  o_hbm, cache_kv = lax.cond(
      num_mixed > 0,
      lambda q_kv: run_mla_kernel(configs.MlaCase.MIXED, q_kv[0], q_kv[1]),
      lambda q_kv: q_kv,
      (ql_nope_prep, cache_kv),
  )

  output = kernel.prepare_outputs(
      o_hbm,
      actual_num_q_heads,
      total_q_tokens,
      actual_lkv_dim,
      vmem_limit_bytes=vmem_limit_bytes,
  )

  return output, cache_kv

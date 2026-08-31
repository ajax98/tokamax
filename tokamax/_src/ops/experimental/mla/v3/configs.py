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

"""Configuration dataclasses, dimensions, and validation for Batched MLA.

This module defines configuration structures and hardware layout rules for
Multi-Head Latent Attention (MLA) on TPU.
"""

import dataclasses
import enum
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.mla.v3 import utils


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  """Tuning block sizes and tiling parameters for the MLA kernel.

  Attributes:
    bq_sz: Query block size (number of sequence tokens per Q block).
    bq_c_sz: Chunked query block size for split execution.
    bkv_sz: KV cache block size (number of context tokens processed per step).
    batch_size: Number of physical TPU batch lanes executing in parallel.
    n_buffer: Pipelining buffer depth (e.g. 2 for double buffering).
  """

  bq_sz: int
  bq_c_sz: int
  bkv_sz: int
  batch_size: int
  n_buffer: int


@dataclasses.dataclass(frozen=True)
class MlaModelConfigs:
  """Immutable architectural parameters of the MLA model.

  MLA decomposes attention representations into:
    - Non-positional latent query/key/value vectors of dimension `lkv_dim`
      (e.g., 512 in DeepSeek-V2/V3).
    - Decoupled rotary embedding (RoPE) query/key vectors of dimension `r_dim`
      (e.g., 64 in DeepSeek-V2/V3).
    - Single compressed KV head (H_kv = 1) shared across all `num_q_heads`
    (e.g., 128).

  Attributes:
    num_q_heads: Number of query attention heads (e.g. 128).
    lkv_dim: Dimension of latent non-positional KV vector d_nope (e.g. 512).
    r_dim: Dimension of decoupled RoPE positional key vector d_pe (e.g. 64).
    mask_value: Large negative value used for causal/padding masking.
    sm_scale: Softmax temperature scale factor (default: 1.0 / sqrt(d_q)).
    soft_cap: Optional logit soft-capping threshold (e.g., Gemma-2 style).
    sliding_window: Optional sliding window attention horizon in tokens.
  """

  num_q_heads: int
  lkv_dim: int
  r_dim: int
  mask_value: float
  sm_scale: float = 1.0
  soft_cap: float | None = None
  sliding_window: int | None = None

  @property
  def total_q_dim(self) -> int:
    """Total query vector dimension d_q = d_nope + d_pe."""
    return self.lkv_dim + self.r_dim

  @property
  def total_kv_dim(self) -> int:
    """Total combined KV vector dimension d_kv = d_nope + d_pe."""
    return self.lkv_dim + self.r_dim


class KVLayout(enum.StrEnum):
  """Memory layout of the paged KV cache in HBM and VMEM.

  - HEAD_ALONG_SUBLANE: Latent dimension is aligned along 128 TPU physical
  lanes;
      sequence tokens are indexed along outer dimensions. Optimal for large
      prefill.
  - SEQ_ALONG_LANE: Sequence tokens are packed along the 128 TPU physical lanes;
      latent dimension is on sublanes. Optimal for autoregressive decode
      (saturates
      128x128 systolic array across query heads when Q_len = 1).
  """

  HEAD_ALONG_SUBLANE = enum.auto()
  SEQ_ALONG_LANE = enum.auto()

  @property
  def symbol(self) -> str:
    match self:
      case KVLayout.HEAD_ALONG_SUBLANE:
        return "nhs"
      case KVLayout.SEQ_ALONG_LANE:
        return "snh"


@dataclasses.dataclass(frozen=True)
class ServingConfigs:
  """Workload and serving batch configuration parameters.

  Attributes:
    num_seqs: Maximum number of active sequences in the current batch.
    page_size: Paged memory page size in tokens (must be multiple of 128 for
      SEQ_ALONG_LANE).
    total_q_tokens: Total flattened query tokens across all sequences (sum of
      q_lens).
    num_page_indices: Size of the flattened page index table (num_seqs *
      pages_per_seq).
    dtype_q: Data type for query tensors (e.g., jnp.bfloat16).
    dtype_kv: Data type for KV cache tensors (e.g., jnp.bfloat16).
    dtype_out: Data type for attention output tensors (e.g., jnp.bfloat16).
    scale_q: Optional scalar quantization multiplier for Q.
    scale_k: Optional scalar quantization multiplier for K.
    scale_v: Optional scalar quantization multiplier for V.
    kv_layout: Paged memory layout (HEAD_ALONG_SUBLANE or SEQ_ALONG_LANE).
    smem_fraction_limit_for_schedule_generation: SMEM budget limit fraction.
    max_schedule_size_multiplier: Multiplier for maximum schedule steps upper
      bound.
  """

  num_seqs: int
  page_size: int
  total_q_tokens: int
  num_page_indices: int
  dtype_q: jnp.dtype
  dtype_kv: jnp.dtype
  dtype_out: jnp.dtype
  scale_q: float | None = None
  scale_k: float | None = None
  scale_v: float | None = None
  kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE
  smem_fraction_limit_for_schedule_generation: float = 0.33
  max_schedule_size_multiplier: int = 16

  @property
  def pages_per_seq(self) -> int:
    return self.num_page_indices // self.num_seqs

  @property
  def page_size_log2(self) -> int:
    return (self.page_size - 1).bit_length()

  @property
  def page_size_mask(self) -> int:
    return self.page_size - 1

  @property
  def packing_q(self) -> int:
    """Number of elements packed per 32-bit word (e.g. 2 for bfloat16)."""
    return utils.get_dtype_packing(self.dtype_q)

  @property
  def packing_kv(self) -> int:
    """Number of elements packed per 32-bit word (e.g. 2 for bfloat16)."""
    return utils.get_dtype_packing(self.dtype_kv)


class MlaCase(enum.StrEnum):
  """Execution mode for the MLA kernel.

  - DECODE: All sequences are in autoregressive decode (q_len = 1).
  - PREFILL: All sequences are in prompt prefill (q_len > 1, static).
  - MIXED: Batch contains a combination of prefill and decode sequences.
  """

  DECODE = enum.auto()
  PREFILL = enum.auto()
  MIXED = enum.auto()

  @property
  def symbol(self) -> str:
    match self:
      case MlaCase.DECODE:
        return "d"
      case MlaCase.PREFILL:
        return "p"
      case MlaCase.MIXED:
        return "m"

  def get_range(self, distribution: Any) -> tuple[Any, Any]:
    """Extracts sequence start and end indices for this execution mode."""
    match self:
      case MlaCase.DECODE:
        return 0, distribution[0]
      case MlaCase.PREFILL:
        return distribution[0], distribution[1]
      case MlaCase.MIXED:
        return distribution[1], distribution[2]


@dataclasses.dataclass(frozen=True, eq=True)
class MlaConfigs:
  """Master configuration combining block sizes, model, serving, and hardware constraints."""

  block: BlockSizes
  model: MlaModelConfigs
  serve: ServingConfigs
  mode: MlaCase
  vmem_limit_bytes: int = 16 * 1024 * 1024

  # Expose block sizes directly for convenient access
  @property
  def bq_sz(self) -> int:
    return self.block.bq_sz

  @property
  def bq_c_sz(self) -> int:
    return self.block.bq_c_sz

  @property
  def bkv_sz(self) -> int:
    return self.block.bkv_sz

  @property
  def batch_size(self) -> int:
    return self.block.batch_size

  @property
  def n_buffer(self) -> int:
    return self.block.n_buffer

  @property
  def q_split(self) -> int:
    """Number of query sub-chunks (bq_sz // bq_c_sz)."""
    return max(1, self.bq_sz // self.bq_c_sz)

  # Derived hardware alignment dimensions
  @property
  def aligned_lkv_dim(self) -> int:
    """d_nope (512) aligned to 128 physical TPU vector lanes."""
    num_lanes = utils.get_tpu_num_lanes()
    return utils.align_to(self.model.lkv_dim, num_lanes)

  @property
  def aligned_r_dim(self) -> int:
    """d_pe (64) aligned to 128 physical TPU vector lanes (aligned to 128)."""
    num_lanes = utils.get_tpu_num_lanes()
    return utils.align_to(self.model.r_dim, num_lanes)

  @property
  def aligned_kv_dim(self) -> int:
    """Combined KV dimension [C_kv, K_pe] aligned to 128-byte multiples."""
    return self.aligned_lkv_dim + self.aligned_r_dim

  @property
  def aligned_q_dim(self) -> int:
    """Combined Query dimension [Q_nope, Q_pe] aligned to 128-byte multiples."""
    return self.aligned_lkv_dim + self.aligned_r_dim

  @property
  def aligned_num_q_heads(self) -> int:
    """Number of Q heads aligned to word-packing boundary."""
    packing_q = self.serve.packing_q
    return utils.align_to(self.model.num_q_heads, packing_q)

  # Paged KV block calculations
  @property
  def bkv_p(self) -> int:
    """Base number of physical pages spanning a single KV block."""
    return pl.cdiv(self.block.bkv_sz, self.serve.page_size)

  @property
  def bkv_p_cache(self) -> int:
    """Number of pages to fetch from the existing cached KV table per step.

    In PREFILL mode, existing cache is not fetched. In DECODE and MIXED, we
    fetch
    at most bkv_p pages (since cached pages are already pre-sliced at offset 0).
    """
    if self.mode == MlaCase.PREFILL:
      return 0
    return self.bkv_p

  @property
  def bkv_p_new(self) -> int:
    """Number of pages to fetch from the unpaged new tokens tensor per step.

    - In DECODE (bq_sz = 1): Exactly 1 new token is decoded, spanning at most 1
    page.
    - In SEQ_ALONG_LANE: Unaligned sequence starts in unpaged HBM can straddle
      across an extra page boundary, requiring (bkv_p + 1) page fetches.
    - In HEAD_ALONG_SUBLANE: Row-contiguous 1D slices do not straddle (bkv_p
    pages).
    """
    if self.mode == MlaCase.DECODE or self.block.bq_sz == 1:
      return 1
    if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
      return self.bkv_p + 1
    return self.bkv_p

  @property
  def dma_kv_new_size(self) -> int:
    """Number of int32 descriptor fields per new-token DMA struct entry.

    - SEQ_ALONG_LANE: 5 fields (fetch_hbm, fetch_vmem, wb_hbm, wb_vmem,
    fetch_val).
    - HEAD_ALONG_SUBLANE: 4 fields (fetch_hbm, fetch_vmem, dst_hbm, fetch_val).
    """
    return 5 if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE else 4

  @property
  def fuse_accum(self) -> bool:
    """Whether to unconditionally normalize acc/l on every block to avoid jax.lax.cond.

    In DECODE (bq_sz = 1), vector division acc / l takes negligible ALU cycles,
    so running it unconditionally eliminates the compiler scheduling barrier.
    In PREFILL (bq_sz >= 64), dividing full matrices is heavy, so we
    conditionally
    execute normalization only on the last block (fuse_accum = False).
    """
    return self.mode == MlaCase.DECODE

  # Per-token byte sizes for DMA lane wait synchronization
  @property
  def kv_bytes_per_token(self) -> int:
    """Byte count transferred per KV token (1 shared latent stream)."""
    return self.aligned_kv_dim * jnp.dtype(self.serve.dtype_kv).itemsize

  @property
  def q_bytes_per_token(self) -> int:
    """Byte count transferred per Query token (H_q heads * (d_nope + d_pe))."""
    return (
        self.aligned_num_q_heads
        * self.aligned_q_dim
        * jnp.dtype(self.serve.dtype_q).itemsize
    )

  @property
  def o_bytes_per_token(self) -> int:
    """Byte count transferred per Output token (H_q heads * d_nope).

    Note: RoPE key is excluded from output; output width is strictly d_nope
    (512).
    """
    return (
        self.aligned_num_q_heads
        * self.aligned_lkv_dim
        * jnp.dtype(self.serve.dtype_out).itemsize
    )

  @property
  def max_steps_ub(self) -> int:
    """Calculates the maximum schedule steps that can fit in TPU SMEM."""
    fixed_bytes = (
        self.serve.num_seqs  # kv_lens
        + (self.serve.num_seqs + 1)  # cu_q_lens
        + (self.serve.num_seqs * self.serve.pages_per_seq)  # page_indices
        + 3  # distribution [decode, prefill, total]
        + self.block.batch_size  # lane_lengths
        + 1  # actual_steps
    ) * 4  # 4 bytes per int32

    smem_limit_bytes = (
        utils.get_tpu_smem_capacity_bytes() - 32 * 1024
    ) * self.serve.smem_fraction_limit_for_schedule_generation
    available_bytes = smem_limit_bytes - fixed_bytes

    bytes_scalars_per_lane = 28
    bytes_cache_dma_per_lane = 12 * self.bkv_p_cache
    bytes_new_dma_per_lane = 4 * self.dma_kv_new_size * self.bkv_p_new
    bytes_global_waits = 16

    bytes_per_step = (
        bytes_scalars_per_lane
        + bytes_cache_dma_per_lane
        + bytes_new_dma_per_lane
    ) * self.block.batch_size + bytes_global_waits

    max_steps_ub = available_bytes // bytes_per_step
    num_lanes = utils.get_tpu_num_lanes()
    return int(max(1, max_steps_ub // num_lanes) * num_lanes)

  # Scratch buffer shapes in VMEM
  @property
  def lm_scratch_shape(self) -> tuple[int, ...]:
    """Scratch shape for running row-max (m) and row-sum (l) vectors."""
    num_lanes = utils.get_tpu_num_lanes()
    return (
        self.block.bq_sz * self.aligned_num_q_heads,
        num_lanes,
    )

  @property
  def acc_scratch_shape(self) -> tuple[int, ...]:
    """Scratch shape for accumulator matrix (width is strictly d_nope = 512)."""
    return (
        self.block.bq_sz * self.aligned_num_q_heads,
        self.aligned_lkv_dim,
    )

  @property
  def kv_vmem_shape(self) -> tuple[int, ...]:
    """VMEM allocation shape for KV buffer [batch_size, sublanes, packing, bkv_sz + 2 * page_size]."""
    num_sublanes = self.aligned_kv_dim // self.serve.packing_kv
    return (
        self.block.batch_size,
        num_sublanes,
        self.serve.packing_kv,
        self.block.bkv_sz + 2 * self.serve.page_size,
    )

  @property
  def q_nope_vmem_shape(self) -> tuple[int, ...]:
    """VMEM allocation shape for non-positional Query buffer [batch_size, bq_sz, words, packing, 128]."""
    q_words = (self.aligned_num_q_heads * self.aligned_lkv_dim) // (
        128 * self.serve.packing_q
    )
    return (
        self.block.batch_size,
        self.block.bq_sz,
        q_words,
        self.serve.packing_q,
        128,
    )

  @property
  def q_pe_vmem_shape(self) -> tuple[int, ...]:
    """VMEM allocation shape for RoPE Query buffer [batch_size, bq_sz, words, packing, 128]."""
    q_pe_words = (self.aligned_num_q_heads * self.aligned_r_dim) // (
        128 * self.serve.packing_q
    )
    return (
        self.block.batch_size,
        self.block.bq_sz,
        q_pe_words,
        self.serve.packing_q,
        128,
    )

  @property
  def o_vmem_shape(self) -> tuple[int, ...]:
    """VMEM allocation shape for output buffer [batch_size, bq_sz, words, packing, 128]."""
    o_words = (self.aligned_num_q_heads * self.aligned_lkv_dim) // (
        128 * self.serve.packing_q
    )
    return (
        self.block.batch_size,
        self.block.bq_sz,
        o_words,
        self.serve.packing_q,
        128,
    )

  @property
  def max_schedule_size_multiplier(self) -> int:
    return self.serve.max_schedule_size_multiplier

  def validate_inputs(
      self,
      ql_nope: jax.Array,
      q_pe: jax.Array,
      new_kv_c: jax.Array,
      new_k_pe: jax.Array,
      cache_kv: jax.Array,
      kv_lens: jax.Array,
      page_indices: jax.Array,
      cu_q_lens: jax.Array,
      distribution: jax.Array,
  ) -> None:
    """Statically validates input shapes, dtypes, and layout constraints."""
    if ql_nope.ndim != 3:
      raise ValueError(
          "Expected 3D array [total_tokens, num_heads, lkv_dim] for ql_nope,"
          f" got {ql_nope.shape}"
      )
    if q_pe.ndim != 3:
      raise ValueError(
          "Expected 3D array [total_tokens, num_heads, r_dim] for q_pe, got"
          f" {q_pe.shape}"
      )
    if new_kv_c.ndim != 2:
      raise ValueError(
          "Expected 2D array [total_tokens, lkv_dim] for new_kv_c, got"
          f" {new_kv_c.shape}"
      )
    if new_k_pe.ndim != 2:
      raise ValueError(
          "Expected 2D array [total_tokens, r_dim] for new_k_pe, got"
          f" {new_k_pe.shape}"
      )

    total_tokens_nope, num_heads_nope, lkv_dim = ql_nope.shape
    total_tokens_pe, num_heads_pe, r_dim = q_pe.shape

    if total_tokens_nope != total_tokens_pe:
      raise ValueError(
          f"Mismatched token count: {total_tokens_nope=} vs {total_tokens_pe=}"
      )
    if num_heads_nope != num_heads_pe:
      raise ValueError(
          f"Mismatched query heads: {num_heads_nope=} vs {num_heads_pe=}"
      )
    if (
        new_kv_c.shape[0] != total_tokens_nope
        or new_k_pe.shape[0] != total_tokens_nope
    ):
      raise ValueError(
          f"Mismatched token count in new KV: {new_kv_c.shape[0]=},"
          f" {new_k_pe.shape[0]=} vs {total_tokens_nope=}"
      )
    if new_kv_c.shape[1] != lkv_dim:
      raise ValueError(
          f"Mismatched latent KV dimension: {new_kv_c.shape[1]=} vs {lkv_dim=}"
      )
    if new_k_pe.shape[1] != r_dim:
      raise ValueError(
          f"Mismatched RoPE key dimension: {new_k_pe.shape[1]=} vs {r_dim=}"
      )

    if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
      if self.serve.page_size % 128 != 0:
        raise ValueError(
            "page_size must be a multiple of 128 for SEQ_ALONG_LANE, got"
            f" {self.serve.page_size=}"
        )
      expected_kv_shape = (
          cache_kv.shape[0],
          self.aligned_kv_dim // self.serve.packing_kv,
          self.serve.packing_kv,
          self.serve.page_size,
      )
    else:
      expected_kv_shape = (
          cache_kv.shape[0],
          self.serve.page_size,
          self.aligned_kv_dim // self.serve.packing_kv,
          self.aligned_kv_dim,
      )

    if cache_kv.shape != expected_kv_shape:
      raise ValueError(
          f"Expected 4D KV cache shape {expected_kv_shape}, got {cache_kv.shape}"
      )

    if not jnp.issubdtype(cache_kv.dtype, jnp.floating):
      raise ValueError(
          f"Expected floating point KV cache, got {cache_kv.dtype}"
      )
    if not (cache_kv.dtype == new_kv_c.dtype == new_k_pe.dtype):
      raise ValueError(
          f"Mismatched dtypes: cache={cache_kv.dtype}, new_c={new_kv_c.dtype},"
          f" new_pe={new_k_pe.dtype}"
      )

    if not (
        jnp.int32
        == kv_lens.dtype
        == page_indices.dtype
        == cu_q_lens.dtype
        == distribution.dtype
    ):
      raise ValueError(
          "Metadata arrays (kv_lens, page_indices, cu_q_lens, distribution)"
          " must be int32."
      )

    max_num_seqs = kv_lens.shape[0]
    if page_indices.shape[0] % max_num_seqs != 0:
      raise ValueError(
          f"page_indices size {page_indices.shape[0]} must be divisible by"
          f" num_seqs {max_num_seqs}"
      )
    if cu_q_lens.shape != (max_num_seqs + 1,):
      raise ValueError(
          f"Expected cu_q_lens shape ({max_num_seqs + 1},), got"
          f" {cu_q_lens.shape}"
      )
    if distribution.shape != (3,):
      raise ValueError(
          f"Expected distribution shape (3,), got {distribution.shape}"
      )

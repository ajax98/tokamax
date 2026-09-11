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

from jax.experimental import pallas as pl
import jax.numpy as jnp
from tokamax._src.ops.experimental.mla.v3 import utils


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  """Tuning block sizes and tiling parameters for the MLA kernel.

  Attributes:
    bq_sz: Query block size (number of sequence tokens per Q block).
    bq_c_sz: Chunked query block size for split execution.
    bkv_sz: KV cache block size (number of context tokens processed per step).
    batch_size: Number of batch lanes packed into one kernel step. The lanes are
      chained *serially* within a step -- lane i's online-softmax state (m, l,
      acc) is rolled forward into lane i+1 -- so this is a work-per-step /
      DMA-overlap knob, not a parallelism knob.
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


class KVLayout(enum.StrEnum):
  """Memory layout of the paged KV cache in HBM and VMEM.

  - SEQ_ALONG_LANE: Sequence tokens are packed along the 128 TPU physical lanes;
      latent dimension is on sublanes. Optimal for autoregressive decode
      (saturates
      128x128 systolic array across query heads when Q_len = 1).

  This is the only layout v3 implements, and the enum exists to name it rather
  than to offer a choice. A `HEAD_ALONG_SUBLANE` member - latent dimension along
  the lanes, sequence tokens on outer dimensions, which would suit large prefill
  - used to sit alongside it, branched for in four files but never finished:
  `dma_kv_new_size` reserved 4 int32 descriptor fields for it while
  `schedule.MlaSchedule.create_shape_dtype` unconditionally allocated the
  5-field `SeqAlongLaneDmaNew`, so `SmemArrayOfStructs.create_shape_dtype`
  asserted before the layout could run at all - and `flash_attention` asserted
  SEQ_ALONG_LANE outright besides. Those branches were deleted rather than left
  to read as a working alternative; git history has them if the layout is ever
  picked up again.
  """

  SEQ_ALONG_LANE = enum.auto()

  @property
  def symbol(self) -> str:
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
    kv_layout: Paged memory layout. Only SEQ_ALONG_LANE is implemented.
    smem_fraction_limit_for_schedule_generation: SMEM budget limit fraction.
    max_schedule_size_multiplier: Floor on the multiplier sizing the HBM
      schedule, in units of `MlaConfigs.max_steps_ub`. Raising it only
      over-allocates; `MlaConfigs.max_schedule_size_multiplier` already raises
      it on its own for any shape that provably needs more.
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
  kv_layout: KVLayout = KVLayout.SEQ_ALONG_LANE
  smem_fraction_limit_for_schedule_generation: float = 0.33
  max_schedule_size_multiplier: int = 16

  # --- Inner-loop precision / scheduling knobs -----------------------------
  #
  # v3 originally had none of these; scores and softmax probabilities stayed
  # f32 throughout and PV followed QK immediately. v2 has all three and its
  # autotuner selected them on every workload measured, which made them the
  # leading explanation for v3's ~1.6x kernel gap on decode_f8_kv9216.
  #
  # Defaults reproduce the original v3 behaviour exactly, so enabling them is
  # opt-in and the existing correctness tests remain a valid baseline.
  #
  # s_dtype: dtype the QK scores are narrowed to before masking and the online
  #   softmax. `None` keeps f32. bf16 is safe for the default mask value
  #   (-0.7 * finfo(f32).max) because bf16 shares f32's 8 exponent bits, so the
  #   sentinel is representable and does not become -inf.
  # p_same_dtype_as_v: cast softmax probabilities to the KV dtype before the PV
  #   matmul, so the MXU sees fp8 x fp8 rather than f32 x fp8. Operand width,
  #   not accumulator width -- accumulation stays f32 either way.
  # two_step_flash_attention: defer a query chunk's PV until after the next
  #   chunk's QK, so VALU (softmax) overlaps MXU (previous PV). Only has an
  #   effect when `q_split > 1`, i.e. `bq_c_sz < bq_sz`.
  s_dtype: Any | None = None
  p_same_dtype_as_v: bool = False
  two_step_flash_attention: bool = False

  # fast_mask: use the single-token mask fast path when a query block holds one
  #   token (`bq_sz == 1`, i.e. all of DECODE). The general path computes
  #   `broadcasted_iota(...) // num_q_heads` over the full [n_q, s_dim] score
  #   tile; with one token per block that expression is identically zero, so
  #   the divide and two of the three full-tile int32 materializations are pure
  #   waste. Off by default so the original path stays the correctness
  #   reference.
  fast_mask: bool = False

  # compact_kv_dim: pad the latent and RoPE parts of the KV dimension to a
  #   whole number of *sublane groups* (packing_kv * 8) rather than to 128
  #   lanes each.
  #
  #   v3 stores the KV dimension on sublanes (SEQ_ALONG_LANE), so 128-lane
  #   alignment is the wrong requirement -- it is inherited from v2, whose
  #   untransposed cache puts kv_dim on lanes and genuinely needs it. For
  #   DeepSeek-V3 the difference is real: align_to(512,128) + align_to(64,128)
  #   = 640, where the true sum is 576. At fp8 that is 144 sublanes, already a
  #   multiple of 8, so 640 buys nothing and costs 10% of every byte of KV --
  #   cache footprint, DMA per block, and layout conversion.
  #
  #   This is a v3-only saving: v2 cannot express it. Off by default because it
  #   changes the cache shape, so it needs a matching arg-spec
  #   (`cache_layout='v3_compact'`).
  compact_kv_dim: bool = False

  # tight_kv_slack: reserve one page of stitch slack instead of two when the
  #   query block holds a single token. Provably sufficient for decode -- see
  #   `MlaConfigs.kv_vmem_lanes` for the bound. Saves 20% of the KV staging
  #   buffer at page_size=1024 and narrows the DMA destination stride.
  tight_kv_slack: bool = False

  # merge_kv_dma: issue one DMA per cache page instead of two. `copy_in` splits
  #   each page into `[:lkv_sublanes]` and `[lkv_sublanes:]` with identical
  #   source and destination lane slices; the ranges are adjacent and together
  #   cover the full sublane extent, so one copy is identical and halves the
  #   descriptor count. Byte totals are unchanged, so the schedule's
  #   `total_wait_kv_in` still matches.
  merge_kv_dma: bool = False

  # disable_bounds_checks: v2 compiles with bounds checks off
  #   (`v2/kernel.py:2635`); v3 leaves them on. With hundreds of DMA
  #   descriptors per step that may not be free.
  disable_bounds_checks: bool = False

  # kv_slack_pad_lanes: extra lanes appended to the KV staging buffer purely to
  #   change its *stride*, not its capacity.
  #
  #   `_stitch_decode_lane` and `store_new_kv_lane` walk the buffer with
  #   `pl.ds(start, outer_dim, lanes_per_col)` where
  #   `lanes_per_col = kv_vmem_lanes // 128`. That stride lands on VMEM banks,
  #   and a power-of-two stride aliases every access onto the same bank.
  #
  #   Measured: shrinking the buffer 5120 -> 4096 lanes (stride 40 -> 32) made
  #   the kernel 49% *slower* despite being strictly less work. 32 is a pure
  #   power of two; 40 is 8x5 and advances through banks. Adding 128 lanes
  #   makes `lanes_per_col` odd (4096+128 -> 33), which is coprime with any
  #   power-of-two bank count and should spread best of all.
  kv_slack_pad_lanes: int = 0

  @property
  def pages_per_seq(self) -> int:
    return self.num_page_indices // self.num_seqs

  @property
  def page_size_log2(self) -> int:
    return (self.page_size - 1).bit_length()

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
  def kv_dim_align(self) -> int:
    """Granularity the KV sub-dimensions are padded to in VMEM: 128 lanes.

    This is the *compute* view and must stay lane-aligned. An earlier attempt
    shrank it to `packing_kv * 8` under `compact_kv_dim`, which also shrank
    `aligned_r_dim` -- and `q_pe` is reshaped with that as its **minor**
    dimension. A TPU vector's minor dim is the lane axis, so Mosaic rejected
    the result:

        tpu.reshape : (vector<1x2x1x16x4x128xf8E4M3FN>) -> vector<2x128x64>
        infer-vector-layout: unsupported shape cast

    The saving is still available, but only on the HBM side: the *cache* keeps
    kv_dim on sublanes, where a sublane group is the real requirement. That
    needs an HBM-compact / VMEM-padded split, not a change here.
    """
    return utils.get_tpu_num_lanes()

  @property
  def aligned_lkv_dim(self) -> int:
    """d_nope (512), padded to 128 lanes."""
    return utils.align_to(self.model.lkv_dim, self.kv_dim_align)

  @property
  def aligned_r_dim(self) -> int:
    """d_pe (64 -> 128), padded to 128 lanes. See `kv_dim_align`."""
    return utils.align_to(self.model.r_dim, self.kv_dim_align)

  @property
  def hbm_r_dim(self) -> int:
    """d_pe as stored in the *HBM* cache.

    Under `compact_kv_dim`, padded only to a sublane group
    (`packing_kv * 8` = 32 at fp8) rather than to 128 lanes: 64 instead of 128
    for DeepSeek-V3. Legal because the cache keeps kv_dim on **sublanes**,
    where whole 32-bit words in groups of 8 is the requirement -- the 128-lane
    rule applies to `q_pe`, which lives on lanes and is left alone (see
    `kv_dim_align`).

    This shrinks the cache and every KV DMA by 10% (640 -> 576) without
    touching the VMEM staging buffer, so `mla_body`, `flash_attention` and the
    stitch stride are all unaffected.
    """
    if self.serve.compact_kv_dim:
      return utils.align_to(self.model.r_dim, self.serve.packing_kv * 8)
    return self.aligned_r_dim

  @property
  def hbm_kv_dim(self) -> int:
    """Total KV width in HBM: 576 compact, 640 otherwise."""
    return self.aligned_lkv_dim + self.hbm_r_dim

  @property
  def hbm_r_sublanes(self) -> int:
    """Sublanes the RoPE part occupies in HBM: 16 compact, 32 otherwise."""
    return self.hbm_r_dim // self.serve.packing_kv

  @property
  def lkv_sublanes(self) -> int:
    """Sublanes the latent part occupies. Same in HBM and VMEM."""
    return self.aligned_lkv_dim // self.serve.packing_kv

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

    At most bkv_p pages, since cached pages are already pre-sliced at offset 0.

    This used to return 0 in PREFILL mode on the assumption that a prefill
    sequence has no history. That only holds for a *whole-prompt* prefill; under
    chunked prefill the earlier chunks are already in the paged cache
    (kv_len > q_len) and skipping the cache fetch silently drops them from the
    attention. PREFILL therefore fetches the cache exactly like MIXED.
    """
    return self.bkv_p

  @property
  def bkv_p_new(self) -> int:
    """Number of pages to fetch from the unpaged new tokens tensor per step.

    - In DECODE (bq_sz = 1): Exactly 1 new token is decoded, spanning at most 1
    page.
    - Otherwise: unaligned sequence starts in unpaged HBM can straddle across an
      extra page boundary, requiring (bkv_p + 1) page fetches.
    """
    if self.mode == MlaCase.DECODE or self.block.bq_sz == 1:
      return 1
    return self.bkv_p + 1

  @property
  def dma_kv_new_size(self) -> int:
    """Number of int32 descriptor fields per new-token DMA struct entry.

    Five, matching `schedule.SeqAlongLaneDmaNew`: fetch_hbm, fetch_vmem, wb_hbm,
    wb_vmem, and the packed flags word.
    """
    return 5

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
    """Byte count transferred per KV token (1 shared latent stream).

    Must be the **HBM** width, not the VMEM width. `_compute_waits` turns this
    into `total_wait_kv_in`, and `KVBufferedRefSeqAlongLane.wait_in` blocks
    until that many bytes have landed. Under `compact_kv_dim` the DMA moves 576
    bytes per token while VMEM is 640 wide -- using the latter here makes the
    wait expect 11% more bytes than are ever sent, and the kernel deadlocks.
    """
    return self.hbm_kv_dim * jnp.dtype(self.serve.dtype_kv).itemsize

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
  def kv_vmem_lanes(self) -> int:
    """Lane extent of the KV staging buffer: `bkv_sz` plus stitch slack.

    The slack holds new-KV pages fetched *past* the cached region before
    `stitch_*_lane` rolls them into place. `fill_dma_kv_new` writes
    `page_size` bytes at `fetch_vmem = (cache_pages + i) * page_size`, so the
    buffer must cover `(cache_pages + i + 1) * page_size`. Two pages in
    general: one for rounding `bkv_sz_cache` up to a page, one for the new
    tokens' own intra-page offset.

    **Decode needs only one.** With `q_len == 1`,
    `new_sz = min(bkv_sz - bkv_sz_cache, kv_left_frm_new) <= 1` token, so
    `num_pages_to_fetch == 1` and only `i = 0` runs -- which the `bq_sz == 1`
    branch in `schedule.fill_dma_kv_new` already asserts via
    `bkv_p_new == 1`. Then `fetch_vmem = cache_pages * page_size <= bkv_sz`
    and the write extends one page: `bkv_sz + page_size` suffices, and the
    second page is never touched.

    Reserving it anyway costs 20% of the buffer at page_size=1024 and widens
    the DMA destination stride (5120 vs 4096 lanes for a 1024-byte write),
    which is why this is worth a flag rather than left as a constant.

    The guard mirrors the condition `stitch_new_kv_lane` uses to select its
    O(1) path, so it cannot apply to a multi-token query block.
    """
    tight = self.serve.tight_kv_slack and (
        self.mode == MlaCase.DECODE or self.block.bq_sz == 1
    )
    return (
        self.block.bkv_sz
        + (1 if tight else 2) * self.serve.page_size
        + self.serve.kv_slack_pad_lanes
    )

  @property
  def kv_vmem_shape(self) -> tuple[int, ...]:
    """VMEM allocation shape for KV buffer [batch_size, sublanes, packing, lanes]."""
    num_sublanes = self.aligned_kv_dim // self.serve.packing_kv
    return (
        self.block.batch_size,
        num_sublanes,
        self.serve.packing_kv,
        self.kv_vmem_lanes,
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
  def max_steps_needed(self) -> int:
    """Upper bound on the schedule steps *any* input of this shape can need.

    The schedule loop increments a counter once per (sequence, q-block, k-block)
    task and packs `batch_size` consecutive tasks into one step, so bounding the
    task count bounds the step count. Both factors below hold for every ragged
    split of `total_q_tokens` across `num_seqs` sequences, which is what makes
    this computable at trace time from shapes alone.

    Nothing here can be tightened by looking at the actual `cu_q_lens` /
    `kv_lens`, because those are runtime values; the bound has to cover the
    worst split the shapes permit. It does assume `kv_len <= pages_per_seq *
    page_size` for every sequence, but a `kv_len` past the end of its page table
    is already an out-of-contract input that the kernel would read the wrong
    pages for.
    """
    # Sequence `s` contributes `cdiv(q_len_s, bq_sz)` q-blocks. Two bounds on
    # the sum are available and neither dominates: `cdiv(q, b) <= q` is tight
    # for decode (many one-token sequences), `cdiv(q, b) <= q // b + 1` is tight
    # for prefill (few long ones).
    q_blocks_ub = min(
        self.serve.total_q_tokens,
        self.serve.num_seqs + self.serve.total_q_tokens // self.block.bq_sz,
    )
    # A q-block visits at most every KV block the page table can address for
    # its sequence. Causal and sliding-window masking only remove blocks from
    # that range, so ignoring both is safe (and, for prefill, loose by ~2x).
    k_blocks_ub = pl.cdiv(
        self.serve.pages_per_seq * self.serve.page_size, self.block.bkv_sz
    )
    return pl.cdiv(q_blocks_ub * k_blocks_ub, self.block.batch_size)

  @property
  def max_schedule_size_multiplier(self) -> int:
    """Sizes the HBM schedule at `max_steps_ub *` this, in steps.

    Raised above the configured value whenever the shape provably needs more
    room, which is what keeps the overflow guard in `_write_schedule_to_hbm`
    from ever firing. Sizing is the fix rather than validation: the bound is a
    worst-case over ragged splits, so rejecting shapes that merely *might*
    overflow would reject shapes that in practice never do, whereas
    over-allocating costs only HBM - a few hundred bytes per step, in a buffer
    the kernel streams rather than resides in.
    """
    return max(
        self.serve.max_schedule_size_multiplier,
        pl.cdiv(self.max_steps_needed, self.max_steps_ub),
    )

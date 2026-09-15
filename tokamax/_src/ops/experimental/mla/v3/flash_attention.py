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

"""Flash Attention compute primitives for Multi-Head Latent Attention (MLA).

Implements online softmax rescaling, QK dot products (non-positional Q_nope *
C_kv +
RoPE Q_pe * K_pe), causal & sliding window masking, and PV accumulation
(accumulating strictly against latent C_kv without RoPE key dimension).
"""

from collections.abc import Sequence
from typing import Any
import jax
from jax import lax
import jax.numpy as jnp
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import utils


def flash_attention_qk_softmax(
    q_fused: jax.Array,  # [B, H_q * T_q, d_nope + d_pe]
    k_fused: jax.Array,  # [B, d_nope + d_pe, S] (SEQ_ALONG_LANE)
    m_prev: jax.Array,  # [H_q * T_q, 128]
    l_prev: jax.Array,  # [H_q * T_q, 128]
    is_last_k: jax.Array | Sequence[Any] | None = None,  # [B]
    *,
    processed_q_len: jax.Array | Sequence[jax.Array] | None = None,  # [B]
    processed_kv_len: jax.Array | Sequence[jax.Array] | None = None,  # [B]
    cfgs: configs.MlaConfigs,
    bq_start: int | jax.Array = 0,
) -> tuple[jax.Array, list[jax.Array], jax.Array, jax.Array]:
  """Computes QK matrix multiplications, masking, and online softmax step for MLA.

  MLA QK score is decomposed into:
    S = sm_scale * (Q_nope @ C_kv.T + Q_pe @ K_pe.T)

  The two contractions are over disjoint *adjacent* slices of the same axis,
  so this is one dot of width `lkv_dim + r_dim` rather than two plus an add.
  The operands arrive pre-concatenated: on the K side that is free, since
  `c_kv` and `k_pe` are already adjacent sublane ranges of `kv_in_vref` and
  their concatenation is the un-sliced ref; only Q needs a real concat.

  Splitting it back into two dots materialised two full [B, H_q, S] f32 score
  tiles -- 6 MB each at the decode optimum -- and read both back to add them.
  That cost 5.6% on decode and 136% on prefill, where the score tile is ~16x
  larger, so the split form is gone rather than configurable.

  Args:
    q_fused: Pre-concatenated `[q_nope | q_pe]`, width `lkv_dim + r_dim`.
    k_fused: Pre-concatenated `[c_kv ; k_pe]`, same width.
    m_prev: Previous running maximum logits for online softmax [H_q * T_q, 128].
    l_prev: Previous running sum of exponentials (denominator) [H_q * T_q, 128].
    is_last_k: Optional boolean flags indicating whether this is the last K
      tile.
    processed_q_len: Sequence start offsets for queries in the batch.
    processed_kv_len: Sequence start offsets for keys in the batch.
    cfgs: MLA configuration parameters.
    bq_start: Block start offset for query within the sequence.

  Returns:
    Tuple of (p, alpha_list, m_carry, l_next).

    `p` is the softmax probability tensor and `alpha_list` the per-batch
    rescaling factors.

    `m_carry` is **not** `m_next`. It is the running max left over after the
    last lane of this block, *after* the end-of-sequence reset: when
    `is_last_k[b]` is set, lane b's carry is forced to `-inf` so the next block
    starts a fresh sequence rather than inheriting the finished one's max. The
    unreset per-lane maxima used to normalize `p` are internal and are not
    returned. `l_next` is the stacked per-lane denominator, and the caller
    keeps only its last lane (`l_next[-1]`) as the carry.
  """
  b = q_fused.shape[0]
  # Rows of the Q block are laid out as `token * aligned_num_q_heads + head`
  # (see `MlaConfigs.q_nope_vmem_shape`), so the row -> token map below must
  # divide by the *padded* head count. Using `model.num_q_heads` skews every
  # token index whenever `num_q_heads` is not a multiple of `packing_q`.
  num_q_heads = cfgs.aligned_num_q_heads
  n_q = q_fused.shape[1]  # aligned_num_q_heads * tq
  assert n_q % num_q_heads == 0, (
      f"Q block rows {n_q} not divisible by aligned head count {num_q_heads}"
  )
  n_tokens = n_q // num_q_heads

  # 1. Compute QK dot products: S = Q_nope @ C_kv.T + Q_pe @ K_pe.T

  s_dim = k_fused.shape[-1]
  s = lax.dot(
      q_fused,
      k_fused,
      dimension_numbers=(([2], [1]), ([0], [0])),
      preferred_element_type=jnp.float32,
  )

  s *= cfgs.model.sm_scale

  if cfgs.serve.scale_k is not None:
    s *= cfgs.serve.scale_k
  if cfgs.serve.scale_q is not None:
    s *= cfgs.serve.scale_q

  # 2. Soft-capping (Gemma / Grok style)
  if cfgs.model.soft_cap is not None:
    s = cfgs.model.soft_cap * jnp.tanh(s / cfgs.model.soft_cap)

  # 3. Causal & Sliding-Window Masking
  if processed_q_len is not None and processed_kv_len is not None:
    sliding_window = cfgs.model.sliding_window

    if n_tokens == 1:
      # One token per block, so q_iota is identically 0 and the predicate
      # collapses to `kv_iota <= -offset`: a [1, s_dim] iota broadcast over
      # rows. The general path below builds a full [n_q, s_dim] iota and an
      # int32 divide per step to produce a tile of zeros. -3.4% on decode.
      kv_iota_1d = lax.broadcasted_iota(jnp.int32, (1, s_dim), 1)
      s_masked = []
      for b_idx in range(b):
        offset = processed_kv_len[b_idx] - (bq_start + processed_q_len[b_idx])
        mask_b = kv_iota_1d <= -offset
        if sliding_window is not None:
          # -kv_iota < sliding_window + offset <=> kv_iota > -offset - sw
          mask_b = jnp.logical_and(
              mask_b, kv_iota_1d > -offset - sliding_window
          )
        s_masked.append(jnp.where(mask_b, s[b_idx], cfgs.model.mask_value))
      s = jnp.stack(s_masked, axis=0)
    else:
      q_iota = lax.broadcasted_iota(jnp.int32, (n_q, s_dim), 0) // num_q_heads
      kv_iota = lax.broadcasted_iota(jnp.int32, (n_q, s_dim), 1)
      q_kv_diff = q_iota - kv_iota

      s_masked = []
      for b_idx in range(b):
        offset = processed_kv_len[b_idx] - (bq_start + processed_q_len[b_idx])
        mask_b = q_kv_diff >= offset

        if sliding_window is not None:
          mask_b = jnp.logical_and(mask_b, q_kv_diff < sliding_window + offset)

        s_masked.append(jnp.where(mask_b, s[b_idx], cfgs.model.mask_value))
      s = jnp.stack(s_masked, axis=0)

  s_curr_max = jnp.max(s, axis=-1, keepdims=True)

  alpha_list = []
  m_next_list = []

  for b_idx in range(b):
    m_curr_b = s_curr_max[b_idx]
    m_next_b = jnp.maximum(m_prev, m_curr_b)
    alpha_b = jnp.exp(m_prev - m_next_b)
    alpha_list.append(alpha_b)
    m_next_list.append(m_next_b)
    if is_last_k is not None:
      m_prev = jnp.where(is_last_k[b_idx], -jnp.inf, m_next_b)
    else:
      m_prev = m_next_b

  m_next = jnp.stack(m_next_list, axis=0)

  # 5. Softmax Probabilities
  #
  # `m_next` is f32 even when `s` is not, so without the cast below the
  # subtract promotes `s` back to f32 and every pass from here on is
  # full-width. See `ServingConfigs.narrow_softmax`.
  p = jnp.exp(s - utils.broadcast_minor(m_next, s.shape))
  p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=jnp.float32)

  l_next_list = []
  for b_idx in range(b):
    l_prev_b = l_prev
    l_next_b = alpha_list[b_idx] * l_prev_b + p_rowsum[b_idx]

    l_next_list.append(l_next_b)
    l_prev = l_next_b

  l_next = jnp.stack(l_next_list, axis=0)

  # `m_prev` here is the post-reset carry for the next block, not `m_next`.
  m_carry = m_prev
  return p, alpha_list, m_carry, l_next


def flash_attention_pv(
    p: jax.Array,  # [B, H_q * T_q, S]
    v: jax.Array,  # [B, d_nope, S]
    alpha_list: Sequence[jax.Array],  # B * [H_q * T_q, 1]
    o_prev: jax.Array,  # [H_q * T_q, d_nope]
    cfgs: configs.MlaConfigs,
) -> jax.Array:
  """Accumulates P @ V where V is strictly latent C_kv (excluding RoPE key K_pe).

  Args:
    p: Softmax probabilities tensor [B, H_q * T_q, S].
    v: Latent KV cache tensor (C_kv) with width d_nope (e.g. 512).
    alpha_list: Per-batch rescaling factor to update previous accumulator.
    o_prev: Previous unnormalized accumulator [H_q * T_q, d_nope].
    cfgs: MLA configuration parameters.

  Returns:
    Updated unnormalized output accumulator [B, H_q * T_q, d_nope].
  """
  b = p.shape[0]

  # Narrows the PV operand, not the accumulator -- `preferred_element_type`
  # below still accumulates in f32. -16.1% on chunked_prefill_f8_kv8192,
  # +9.4% on decode, so it stays per-shape.
  if cfgs.serve.p_same_dtype_as_v:
    p = p.astype(v.dtype)

  pv = lax.dot(
      p,
      v,
      dimension_numbers=(([2], [2]), ([0], [0])),
      preferred_element_type=jnp.float32,
  )
  if cfgs.serve.scale_v is not None:
    pv *= cfgs.serve.scale_v

  o_next_list = []
  for b_idx in range(b):
    scaled = utils.broadcast_minor(alpha_list[b_idx], o_prev.shape) * o_prev
    o_next_b = scaled + pv[b_idx]
    o_next_list.append(o_next_b)
    o_prev = o_next_b

  return jnp.stack(o_next_list, axis=0)


def chunked_flash_attention(
    q_fused: jax.Array,  # [B, H_q * bq_sz, d_nope + d_pe]
    k_fused: jax.Array,  # [B, d_nope + d_pe, S]
    k_nope: jax.Array,  # [B, d_nope, S] -- PV only, narrower than k_fused
    m_prev: jax.Array,  # [H_q * bq_sz, 128]
    l_prev: jax.Array,  # [H_q * bq_sz, 128]
    o_prev: jax.Array,  # [H_q * bq_sz, d_nope]
    is_last_k: jax.Array | Sequence[Any] | None = None,  # [B]
    *,
    processed_q_len: jax.Array | Sequence[jax.Array] | None = None,
    processed_kv_len: jax.Array | Sequence[jax.Array] | None = None,
    cfgs: configs.MlaConfigs,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Executes Flash Attention chunked into bq_c_sz sub-blocks along the query dimension.

  Splits the query tensor of size bq_sz into q_split sub-chunks of size bq_c_sz
  (q_split = bq_sz // bq_c_sz) to reduce VMEM register pressure during GEMM
  execution.

  Args:
    q_fused: Pre-concatenated `[q_nope | q_pe]`, sliced per chunk for QK.
    k_fused: Pre-concatenated `[c_kv ; k_pe]`, used for QK.
    k_nope: Latent KV cache (C_kv), width d_nope. PV only -- it accumulates
      against the narrower operand, so this is not the same array as
      `k_fused`.
    m_prev: Previous running maximum logits [H_q * bq_sz, 128].
    l_prev: Previous running denominator [H_q * bq_sz, 128].
    o_prev: Previous unnormalized accumulator [H_q * bq_sz, d_nope].
    is_last_k: Optional boolean flags indicating whether this is the last K
      tile.
    processed_q_len: Sequence start offsets for queries in the batch.
    processed_kv_len: Sequence start offsets for keys in the batch.
    cfgs: MLA configuration parameters (containing bq_sz and bq_c_sz).

  Returns:
    Tuple of (m_carry, l_next, o_next).

    Note the asymmetry, inherited from `flash_attention_qk_softmax`: `m_carry`
    is rank-2 -- the single post-reset running max left after the *last* batch
    lane -- whereas `l_next` and `o_next` are stacked per-lane and carry a
    leading [B] axis. Callers therefore store `m_carry` directly but must take
    `l_next[-1]` / `o_next[-1]` to get the corresponding carries.
  """
  # Row chunking divides the *token* axis (`q_split`, from `bq_sz // bq_c_sz`):
  # each chunk is a run of whole tokens and carries its own `bq_start` offset
  # into the mask.
  q_split = cfgs.q_split
  if q_split == 1:
    p, alpha_list, m_carry, l_next = flash_attention_qk_softmax(
        q_fused,
        k_fused,
        m_prev,
        l_prev,
        is_last_k=is_last_k,
        processed_q_len=processed_q_len,
        processed_kv_len=processed_kv_len,
        cfgs=cfgs,
        bq_start=0,
    )
    o_next = flash_attention_pv(p, k_nope, alpha_list, o_prev, cfgs=cfgs)
    return m_carry, l_next, o_next

  total_q = q_fused.shape[1]
  assert total_q % q_split == 0, (
      f"Q block rows {total_q} not divisible by {q_split=}"
  )
  q_chunk_len = total_q // q_split
  bq_sz_chunk = cfgs.bq_c_sz

  m_carry_splits = []
  l_next_splits = []
  o_next_splits = []

  # A chunk's PV is deferred until the next chunk's QK is issued, so MXU
  # and VALU overlap instead of idling in turn. Inert at q_split == 1.
  pending = None  # (p, alpha, o_prev, slot)

  def _flush(pending):
    p_chunk, alpha_chunk, o_prev_chunk, slot = pending
    o_next_splits[slot] = flash_attention_pv(
        p_chunk, k_nope, alpha_chunk, o_prev_chunk, cfgs=cfgs
    )

  for q_idx in range(q_split):
    start = q_idx * q_chunk_len
    end = start + q_chunk_len
    bq_start = q_idx * bq_sz_chunk

    q_fused_chunk = q_fused[:, start:end]
    m_prev_chunk = m_prev[start:end]
    l_prev_chunk = l_prev[start:end]
    o_prev_chunk = o_prev[start:end]

    p_chunk, alpha_chunk, m_carry_chunk, l_next_chunk = (
        flash_attention_qk_softmax(
            q_fused_chunk,
            k_fused,
            m_prev_chunk,
            l_prev_chunk,
            is_last_k=is_last_k,
            processed_q_len=processed_q_len,
            processed_kv_len=processed_kv_len,
            cfgs=cfgs,
            bq_start=bq_start,
        )
    )

    # Retire the previous chunk's PV now that this chunk's QK is in flight.
    if pending is not None:
      _flush(pending)
    o_next_splits.append(None)  # placeholder, filled by `_flush`
    pending = (p_chunk, alpha_chunk, o_prev_chunk, q_idx)
    m_carry_splits.append(m_carry_chunk)
    l_next_splits.append(l_next_chunk)

  if pending is not None:
    _flush(pending)
  assert all(o is not None for o in o_next_splits)

  return (
      # `m_carry` chunks tile the query axis (axis 0); `l`/`o` chunks tile the
      # query axis of a [B, ...] stack, hence axis 1.
      jnp.concatenate(m_carry_splits, axis=0),
      jnp.concatenate(l_next_splits, axis=1),
      jnp.concatenate(o_next_splits, axis=1),
  )

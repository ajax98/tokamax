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
    q_nope: jax.Array,  # [B, H_q * T_q, d_nope]
    q_pe: jax.Array,  # [B, H_q * T_q, d_pe]
    k_nope: jax.Array,  # [B, d_nope, S] (SEQ_ALONG_LANE)
    k_pe: jax.Array,  # [B, d_pe, S] (SEQ_ALONG_LANE)
    m_prev: jax.Array,  # [H_q * T_q, 128]
    l_prev: jax.Array,  # [H_q * T_q, 128]
    is_last_k: jax.Array | Sequence[Any] | None = None,  # [B]
    *,
    processed_q_len: jax.Array | Sequence[jax.Array] | None = None,  # [B]
    processed_kv_len: jax.Array | Sequence[jax.Array] | None = None,  # [B]
    cfgs: configs.MlaConfigs,
    bq_start: int | jax.Array = 0,
    q_fused: jax.Array | None = None,  # [B, H_q * T_q, d_nope + d_pe]
    k_fused: jax.Array | None = None,  # [B, d_nope + d_pe, S]
    n_tokens: int | None = None,
) -> tuple[jax.Array, list[jax.Array], jax.Array, jax.Array]:
  """Computes QK matrix multiplications, masking, and online softmax step for MLA.

  MLA QK score is decomposed into:
    S = sm_scale * (Q_nope @ C_kv.T + Q_pe @ K_pe.T)

  Args:
    q_nope: Non-positional query tensor of width lkv_dim (e.g. 512).
    q_pe: Decoupled RoPE query tensor of width r_dim (e.g. 64).
    k_nope: Latent KV cache tensor (C_kv) of width lkv_dim.
    k_pe: Decoupled RoPE key cache tensor (K_pe) of width r_dim.
    m_prev: Previous running maximum logits for online softmax [H_q * T_q, 128].
    l_prev: Previous running sum of exponentials (denominator) [H_q * T_q, 128].
    is_last_k: Optional boolean flags indicating whether this is the last K
      tile.
    processed_q_len: Sequence start offsets for queries in the batch.
    processed_kv_len: Sequence start offsets for keys in the batch.
    cfgs: MLA configuration parameters.
    bq_start: Block start offset for query within the sequence.
    q_fused: Optional pre-concatenated `[q_nope | q_pe]`. When supplied along
      with `k_fused`, the two QK dots below collapse into one; `q_nope` and
      `q_pe` are then used only for their shapes.
    k_fused: Optional pre-concatenated `[c_kv ; k_pe]`.
    n_tokens: Number of query tokens in this block. Normally derived as
      `n_q // aligned_num_q_heads`, but under `head_split` a chunk holds a
      *subset of one token's heads*, so it has fewer rows than
      `aligned_num_q_heads` and the quotient would be 0. Callers that chunk
      the head axis pass the true count explicitly.

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
  b = q_nope.shape[0]
  # Rows of the Q block are laid out as `token * aligned_num_q_heads + head`
  # (see `MlaConfigs.q_nope_vmem_shape`), so the row -> token map below must
  # divide by the *padded* head count. Using `model.num_q_heads` skews every
  # token index whenever `num_q_heads` is not a multiple of `packing_q`.
  num_q_heads = cfgs.aligned_num_q_heads
  n_q = q_nope.shape[1]  # aligned_num_q_heads * tq, or a head-axis slice
  if n_tokens is None:
    assert n_q % num_q_heads == 0, (
        f"Q block rows {n_q} not divisible by aligned head count {num_q_heads}"
    )
    n_tokens = n_q // num_q_heads

  # 1. Compute QK dot products: S = Q_nope @ C_kv.T + Q_pe @ K_pe.T

  # k_nope: [b, d_nope, s], k_pe: [b, d_pe, s]
  s_dim = k_nope.shape[-1]
  if q_fused is not None and k_fused is not None:
    # Single contraction over the concatenated [nope | pe] axis. Identical
    # arithmetic to the two-dot form below, but only one f32 score tile is
    # ever live.
    s = lax.dot(
        q_fused,
        k_fused,
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
  else:
    s_nope = lax.dot(
        q_nope,
        k_nope,
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
    s_pe = lax.dot(
        q_pe,
        k_pe,
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    )
    s = s_nope + s_pe

  s *= cfgs.model.sm_scale

  if cfgs.serve.scale_k is not None:
    s *= cfgs.serve.scale_k
  if cfgs.serve.scale_q is not None:
    s *= cfgs.serve.scale_q

  # 2. Soft-capping (Gemma / Grok style)
  if cfgs.model.soft_cap is not None:
    s = cfgs.model.soft_cap * jnp.tanh(s / cfgs.model.soft_cap)

  # 2b. Optionally narrow the score tile before masking and softmax.
  #
  # `s` is [B, H_q * T_q, S] -- the widest intermediate in the kernel -- and
  # everything downstream of here (the mask `jnp.where`, the row max, the
  # `exp`, the row sum) is vector work over it. Narrowing to bf16 halves that
  # VREG traffic. This mirrors `v2/kernel.py:661`, which casts to `s_dtype`
  # (bf16 by default) at exactly this point.
  #
  # Note the QK dot above already accumulated in f32, and `p` below comes back
  # out as f32 because `m` is f32; the saving is in the vector ops, not the
  # matmul.
  # 3. Causal & Sliding-Window Masking
  if processed_q_len is not None and processed_kv_len is not None:
    sliding_window = cfgs.model.sliding_window

    if n_tokens == 1:
      # Single-token fast path (decode, and any bq_sz==1 block).
      #
      # Rows are laid out `token * aligned_num_q_heads + head`, so
      # `row // num_q_heads` is the token index -- and with one token per block
      # it is identically 0. The general path below therefore materializes a
      # full [n_q, s_dim] iota and runs an int32 *divide* over it, every step,
      # to produce a tile of zeros. At the decode optimum that tile is
      # [128, 2048]: ~1 MB, three times over (two iotas and their difference),
      # plus the divide.
      #
      # With q_iota == 0 the predicate collapses to a 1-D comparison against
      # the KV lane index:
      #     q_iota - kv_iota >= offset   <=>   kv_iota <= -offset
      # so a [1, s_dim] iota broadcast over rows is sufficient. No divide, no
      # 2-D query iota, no full-tile subtract.
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

  return _finish_softmax(s, b, m_prev, l_prev, is_last_k, cfgs=cfgs)


def _finish_softmax(
    s: jax.Array,  # [B, H_q * T_q, S], already masked
    b: int,
    m_prev: jax.Array,
    l_prev: jax.Array,
    is_last_k: jax.Array | Sequence[Any] | None,
    *,
    cfgs: configs.MlaConfigs,
) -> tuple[jax.Array, list[jax.Array], jax.Array, jax.Array]:
  """Online-softmax tail shared by every masking path.

  Split out so the `tiled_mask` path can reuse it verbatim instead of
  duplicating the lane-chaining, which is the subtle part: lanes are
  consecutive schedule tasks sharing one carry, and `is_last_k` resets the
  chain at sequence boundaries.
  """
  # 4. Online Softmax Running Statistics
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

  # Narrow the PV *operand*, not the accumulator.
  #
  # `preferred_element_type=jnp.float32` below sets the accumulation width and
  # is unchanged; what this cast changes is what the MXU is fed. Without it
  # `p` arrives as f32 (out of `jnp.exp`) and `v` is the KV dtype, so the dot
  # runs at the wider operand's rate. v2 does the same thing at
  # `v2/kernel.py:733` under `p_same_dtype_as_v`, which its autotuner selected
  # on every workload measured. At FP8 the gap between fp8 x fp8 and f32 x fp8
  # is large, and PV is roughly half the kernel's FLOPs.
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
    q_nope: jax.Array,  # [B, H_q * bq_sz, d_nope]
    q_pe: jax.Array,  # [B, H_q * bq_sz, d_pe]
    k_nope: jax.Array,  # [B, d_nope, S]
    k_pe: jax.Array,  # [B, d_pe, S]
    m_prev: jax.Array,  # [H_q * bq_sz, 128]
    l_prev: jax.Array,  # [H_q * bq_sz, 128]
    o_prev: jax.Array,  # [H_q * bq_sz, d_nope]
    is_last_k: jax.Array | Sequence[Any] | None = None,  # [B]
    *,
    processed_q_len: jax.Array | Sequence[jax.Array] | None = None,
    processed_kv_len: jax.Array | Sequence[jax.Array] | None = None,
    cfgs: configs.MlaConfigs,
    q_fused: jax.Array | None = None,  # [B, H_q * bq_sz, d_nope + d_pe]
    k_fused: jax.Array | None = None,  # [B, d_nope + d_pe, S]
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Executes Flash Attention chunked into bq_c_sz sub-blocks along the query dimension.

  Splits the query tensor of size bq_sz into q_split sub-chunks of size bq_c_sz
  (q_split = bq_sz // bq_c_sz) to reduce VMEM register pressure during GEMM
  execution.

  Args:
    q_nope: Full query non-positional tensor [B, H_q * bq_sz, d_nope].
    q_pe: Full query RoPE tensor [B, H_q * bq_sz, d_pe].
    k_nope: Latent KV cache tensor (C_kv) of width d_nope.
    k_pe: Decoupled RoPE key cache tensor (K_pe) of width d_pe.
    m_prev: Previous running maximum logits [H_q * bq_sz, 128].
    l_prev: Previous running denominator [H_q * bq_sz, 128].
    o_prev: Previous unnormalized accumulator [H_q * bq_sz, d_nope].
    is_last_k: Optional boolean flags indicating whether this is the last K
      tile.
    processed_q_len: Sequence start offsets for queries in the batch.
    processed_kv_len: Sequence start offsets for keys in the batch.
    cfgs: MLA configuration parameters (containing bq_sz and bq_c_sz).
    q_fused: Optional pre-concatenated `[q_nope | q_pe]`, forwarded to
      `flash_attention_qk_softmax` and sliced per query chunk alongside
      `q_nope`.
    k_fused: Optional pre-concatenated `[c_kv ; k_pe]`. Used for QK only --
      `flash_attention_pv` keeps accumulating against the narrower `k_nope`.

  Returns:
    Tuple of (m_carry, l_next, o_next).

    Note the asymmetry, inherited from `flash_attention_qk_softmax`: `m_carry`
    is rank-2 -- the single post-reset running max left after the *last* batch
    lane -- whereas `l_next` and `o_next` are stacked per-lane and carry a
    leading [B] axis. Callers therefore store `m_carry` directly but must take
    `l_next[-1]` / `o_next[-1]` to get the corresponding carries.
  """
  # Row chunking serves two different splits. The original one divides the
  # *token* axis (`q_split`, from `bq_sz // bq_c_sz`), where each chunk is a
  # run of whole tokens and carries its own `bq_start` offset into the mask.
  # `head_split` instead divides the *head* axis within a single token, so
  # every chunk sits at token 0 and `bq_start` stays 0. The two never combine:
  # head_split only engages at `bq_sz == 1`, where `q_split` is already 1.
  # See `ServingConfigs.head_split`.
  q_split = cfgs.q_split
  if q_split == 1:
    p, alpha_list, m_carry, l_next = flash_attention_qk_softmax(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        m_prev,
        l_prev,
        is_last_k=is_last_k,
        processed_q_len=processed_q_len,
        processed_kv_len=processed_kv_len,
        cfgs=cfgs,
        bq_start=0,
        q_fused=q_fused,
        k_fused=k_fused,
    )
    o_next = flash_attention_pv(p, k_nope, alpha_list, o_prev, cfgs=cfgs)
    return m_carry, l_next, o_next

  total_q = q_nope.shape[1]
  assert total_q % q_split == 0, (
      f"Q block rows {total_q} not divisible by {q_split=}"
  )
  q_chunk_len = total_q // q_split
  # Under `head_split` every chunk is part of the same single token, so the
  # per-chunk mask offset is 0 and the token count is 1 rather than
  # `q_chunk_len // aligned_num_q_heads` (which would be 0 for a sub-head-count
  # chunk).
  bq_sz_chunk = cfgs.bq_c_sz

  m_carry_splits = []
  l_next_splits = []
  o_next_splits = []

  # With `two_step_flash_attention`, a chunk's PV is deferred until after the
  # *next* chunk's QK+softmax has been issued, so the MXU work of the former
  # overlaps the VALU work of the latter. Without it, each chunk runs QK then
  # PV back-to-back and the two units idle in turn.
  #
  # This mirrors `v2/kernel.py:1986-2026`, which keeps `prev_p`/`prev_v` across
  # loop iterations and flushes the final PV after the loop. Note it only bites
  # when `q_split > 1` -- with a single chunk there is no next QK to hide
  # behind, which is the case for DECODE (`bq_sz = 1`).
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

    q_nope_chunk = q_nope[:, start:end]
    q_pe_chunk = q_pe[:, start:end]
    q_fused_chunk = None if q_fused is None else q_fused[:, start:end]
    m_prev_chunk = m_prev[start:end]
    l_prev_chunk = l_prev[start:end]
    o_prev_chunk = o_prev[start:end]

    p_chunk, alpha_chunk, m_carry_chunk, l_next_chunk = (
        flash_attention_qk_softmax(
            q_nope_chunk,
            q_pe_chunk,
            k_nope,
            k_pe,
            m_prev_chunk,
            l_prev_chunk,
            is_last_k=is_last_k,
            processed_q_len=processed_q_len,
            processed_kv_len=processed_kv_len,
            cfgs=cfgs,
            q_fused=q_fused_chunk,
            k_fused=k_fused,
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

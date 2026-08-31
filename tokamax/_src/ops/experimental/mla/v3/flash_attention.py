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
from jax.experimental import pallas as pl
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

  Returns:
    Tuple of (p, alpha_list, m_next, l_next) where p is softmax probability
    tensor, alpha_list contains per-batch rescaling factors, m_next is updated
    running max, and l_next is updated denominator.
  """
  b = q_nope.shape[0]
  num_q_heads = cfgs.model.num_q_heads
  n_q = q_nope.shape[1]  # num_q_heads * tq

  # Consider supporting HEAD_ALONG_SUBLANE layout later.
  assert cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE

  # 1. Compute QK dot products: S = Q_nope @ C_kv.T + Q_pe @ K_pe.T

  # k_nope: [b, d_nope, s], k_pe: [b, d_pe, s]
  s_dim = k_nope.shape[-1]
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

  # 3. Causal & Sliding-Window Masking
  if processed_q_len is not None and processed_kv_len is not None:
    q_iota = lax.broadcasted_iota(jnp.int32, (n_q, s_dim), 0) // num_q_heads
    kv_iota = lax.broadcasted_iota(jnp.int32, (n_q, s_dim), 1)
    q_kv_diff = q_iota - kv_iota

    s_masked = []
    for b_idx in range(b):
      offset = processed_kv_len[b_idx] - (bq_start + processed_q_len[b_idx])
      mask_b = q_kv_diff >= offset

      if (sliding_window := cfgs.model.sliding_window) is not None:
        mask_b = jnp.logical_and(mask_b, q_kv_diff < sliding_window + offset)

      s_masked.append(jnp.where(mask_b, s[b_idx], cfgs.model.mask_value))
    s = jnp.stack(s_masked, axis=0)

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
  p = jnp.exp(s - utils.broadcast_minor(m_next, s.shape))
  p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=jnp.float32)

  l_next_list = []
  for b_idx in range(b):
    l_prev_b = l_prev
    l_next_b = alpha_list[b_idx] * l_prev_b + p_rowsum[b_idx]

    l_next_list.append(l_next_b)
    l_prev = l_next_b

  l_next = jnp.stack(l_next_list, axis=0)

  return p, alpha_list, m_prev, l_next


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
  b, n_q, s = p.shape

  assert cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE
  d_nope = v.shape[-2]
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
    alpha_b = utils.broadcast_minor(alpha_list[b_idx], o_prev.shape)
    o_next_b = alpha_b * o_prev + pv[b_idx]
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

  Returns:
    Tuple of (m_next, l_next, o_next) containing updated running statistics and
    accumulated outputs across all query sub-chunks.
  """
  q_split = cfgs.q_split
  if q_split == 1:
    p, alpha_list, m_next, l_next = flash_attention_qk_softmax(
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
    )
    o_next = flash_attention_pv(p, k_nope, alpha_list, o_prev, cfgs=cfgs)
    return m_next, l_next, o_next

  total_q = q_nope.shape[1]
  q_chunk_len = total_q // q_split
  bq_sz_chunk = cfgs.bq_c_sz

  m_next_splits = []
  l_next_splits = []
  o_next_splits = []

  for q_idx in range(q_split):
    start = q_idx * q_chunk_len
    end = start + q_chunk_len
    bq_start = q_idx * bq_sz_chunk

    q_nope_chunk = q_nope[:, start:end]
    q_pe_chunk = q_pe[:, start:end]
    m_prev_chunk = m_prev[start:end]
    l_prev_chunk = l_prev[start:end]
    o_prev_chunk = o_prev[start:end]

    p_chunk, alpha_chunk, m_next_chunk, l_next_chunk = (
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
            bq_start=bq_start,
        )
    )
    o_next_chunk = flash_attention_pv(
        p_chunk, k_nope, alpha_chunk, o_prev_chunk, cfgs=cfgs
    )

    m_next_splits.append(m_next_chunk)
    l_next_splits.append(l_next_chunk)
    o_next_splits.append(o_next_chunk)

  return (
      jnp.concatenate(m_next_splits, axis=0),
      jnp.concatenate(l_next_splits, axis=1),
      jnp.concatenate(o_next_splits, axis=1),
  )

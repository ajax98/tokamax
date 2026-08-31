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

"""Unit tests for MLA Flash Attention pure math engine."""

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import flash_attention
from tokamax._src.ops.experimental.mla.v3 import utils


def _normalize_output(
    acc: jax.Array,
    l: jax.Array,
    *,
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
  """Normalizes unnormalized attention accumulator by denominator l."""
  l_broadcasted = utils.broadcast_minor(l, acc.shape)
  if out_dtype == jnp.float32:
    return lax.div(acc, l_broadcasted)
  return (acc * pl.reciprocal(l_broadcasted, approx=True)).astype(out_dtype)


class FlashAttentionMathTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 1
    self.num_q_heads = 4
    self.lkv_dim = 512
    self.r_dim = 64
    self.sm_scale = 1.0 / np.sqrt(self.lkv_dim + self.r_dim)

    self.model_cfg = configs.MlaModelConfigs(
        num_q_heads=self.num_q_heads,
        lkv_dim=self.lkv_dim,
        r_dim=self.r_dim,
        sm_scale=self.sm_scale,
        mask_value=-1e9,
    )

  def test_decode_flash_attention_numerical_correctness(self):
    """Verifies decode (Bq=1) MLA attention math matches reference JAX einsum."""
    b, bq_sz, s = self.batch_size, 1, 128
    block_cfg = configs.BlockSizes(
        bq_sz=bq_sz, bq_c_sz=bq_sz, bkv_sz=s, batch_size=b, n_buffer=2
    )
    serving_cfg = configs.ServingConfigs(
        num_seqs=b,
        page_size=s,
        total_q_tokens=b * bq_sz,
        num_page_indices=1,
        dtype_q=jnp.float32,
        dtype_kv=jnp.float32,
        dtype_out=jnp.float32,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    mla_cfg = configs.MlaConfigs(
        block=block_cfg,
        model=self.model_cfg,
        serve=serving_cfg,
        mode=configs.MlaCase.DECODE,
    )

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Q: [B, H_q * T_q, D]
    n_q = self.num_q_heads * bq_sz
    q_nope = jax.random.normal(k1, (b, n_q, self.lkv_dim), dtype=jnp.float32)
    q_pe = jax.random.normal(k2, (b, n_q, self.r_dim), dtype=jnp.float32)

    # K/V in SEQ_ALONG_LANE: [B, D, S]
    k_nope = jax.random.normal(k3, (b, self.lkv_dim, s), dtype=jnp.float32)
    k_pe = jax.random.normal(k4, (b, self.r_dim, s), dtype=jnp.float32)

    # Reference computation:
    # S = sm_scale * (q_nope @ k_nope + q_pe @ k_pe)
    s_ref = self.sm_scale * (
        jnp.einsum("bnd,bds->bns", q_nope, k_nope)
        + jnp.einsum("bnd,bds->bns", q_pe, k_pe)
    )
    p_ref = jax.nn.softmax(s_ref, axis=-1)
    # V is strictly k_nope (C_kv)
    out_ref = jnp.einsum("bns,bds->bnd", p_ref, k_nope)

    # Flash attention compute (matching Pallas kernel scratch shapes)
    m_prev = jnp.full((n_q, 128), -jnp.inf, dtype=jnp.float32)
    l_prev = jnp.zeros((n_q, 128), dtype=jnp.float32)
    o_prev = jnp.zeros((n_q, self.lkv_dim), dtype=jnp.float32)

    p, alpha_list, m_next, l_next = flash_attention.flash_attention_qk_softmax(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        m_prev,
        l_prev,
        cfgs=mla_cfg,
    )
    acc = flash_attention.flash_attention_pv(
        p, k_nope, alpha_list, o_prev, cfgs=mla_cfg
    )
    out_flash = _normalize_output(
        acc, l_next, out_dtype=jnp.float32
    )

    np.testing.assert_allclose(out_flash, out_ref, rtol=1e-4, atol=1e-4)

  def test_prefill_multi_block_online_softmax_accumulation(self):
    """Verifies online softmax state accumulation across 2 sequential K blocks."""
    b, bq_sz, s = 1, 4, 64
    block_cfg = configs.BlockSizes(
        bq_sz=bq_sz, bq_c_sz=bq_sz, bkv_sz=s, batch_size=b, n_buffer=2
    )
    serving_cfg = configs.ServingConfigs(
        num_seqs=b,
        page_size=s,
        total_q_tokens=b * bq_sz,
        num_page_indices=2,
        dtype_q=jnp.float32,
        dtype_kv=jnp.float32,
        dtype_out=jnp.float32,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    mla_cfg = configs.MlaConfigs(
        block=block_cfg,
        model=self.model_cfg,
        serve=serving_cfg,
        mode=configs.MlaCase.PREFILL,
    )

    key = jax.random.PRNGKey(101)
    keys = jax.random.split(key, 6)

    n_q = self.num_q_heads * bq_sz
    q_nope = jax.random.normal(keys[0], (b, n_q, self.lkv_dim), dtype=jnp.float32)
    q_pe = jax.random.normal(keys[1], (b, n_q, self.r_dim), dtype=jnp.float32)

    # Block 0 K/V
    k0_nope = jax.random.normal(keys[2], (b, self.lkv_dim, s), dtype=jnp.float32)
    k0_pe = jax.random.normal(keys[3], (b, self.r_dim, s), dtype=jnp.float32)

    # Block 1 K/V
    k1_nope = jax.random.normal(keys[4], (b, self.lkv_dim, s), dtype=jnp.float32)
    k1_pe = jax.random.normal(keys[5], (b, self.r_dim, s), dtype=jnp.float32)

    # Reference across combined [K0, K1] (total length 2 * s = 128)
    k_all_nope = jnp.concatenate([k0_nope, k1_nope], axis=-1)
    k_all_pe = jnp.concatenate([k0_pe, k1_pe], axis=-1)

    s_all = self.sm_scale * (
        jnp.einsum("bnd,bds->bns", q_nope, k_all_nope)
        + jnp.einsum("bnd,bds->bns", q_pe, k_all_pe)
    )
    p_all = jax.nn.softmax(s_all, axis=-1)
    out_ref = jnp.einsum("bns,bds->bnd", p_all, k_all_nope)

    # Iteration 0 (Block 0)
    m0 = jnp.full((n_q, 128), -jnp.inf, dtype=jnp.float32)
    l0 = jnp.zeros((n_q, 128), dtype=jnp.float32)
    o0 = jnp.zeros((n_q, self.lkv_dim), dtype=jnp.float32)

    p0, alpha0, m1, l1 = flash_attention.flash_attention_qk_softmax(
        q_nope, q_pe, k0_nope, k0_pe, m0, l0, cfgs=mla_cfg
    )
    acc1 = flash_attention.flash_attention_pv(
        p0, k0_nope, alpha0, o0, cfgs=mla_cfg
    )

    # Iteration 1 (Block 1) - matches kernel VMEM scratch state [n_q, 128] / [n_q, d_nope]
    p1, alpha1, m2, l2 = flash_attention.flash_attention_qk_softmax(
        q_nope, q_pe, k1_nope, k1_pe, m1, l1[-1], cfgs=mla_cfg
    )
    acc2 = flash_attention.flash_attention_pv(
        p1, k1_nope, alpha1, acc1[-1], cfgs=mla_cfg
    )

    # Normalize final output
    out_flash = _normalize_output(
        acc2, l2, out_dtype=jnp.float32
    )

    np.testing.assert_allclose(out_flash, out_ref, rtol=1e-4, atol=1e-4)

  def test_chunked_query_splitting(self):
    """Verifies that splitting bq_sz into bq_c_sz sub-chunks produces identical results."""
    b, bq_sz, bq_c_sz, s = 1, 8, 2, 64  # q_split = 4
    block_cfg = configs.BlockSizes(
        bq_sz=bq_sz, bq_c_sz=bq_c_sz, bkv_sz=s, batch_size=b, n_buffer=2
    )
    serving_cfg = configs.ServingConfigs(
        num_seqs=b,
        page_size=s,
        total_q_tokens=b * bq_sz,
        num_page_indices=1,
        dtype_q=jnp.float32,
        dtype_kv=jnp.float32,
        dtype_out=jnp.float32,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    mla_cfg = configs.MlaConfigs(
        block=block_cfg,
        model=self.model_cfg,
        serve=serving_cfg,
        mode=configs.MlaCase.PREFILL,
    )

    self.assertEqual(mla_cfg.q_split, 4)

    key = jax.random.PRNGKey(999)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    n_q = self.num_q_heads * bq_sz
    q_nope = jax.random.normal(k1, (b, n_q, self.lkv_dim), dtype=jnp.float32)
    q_pe = jax.random.normal(k2, (b, n_q, self.r_dim), dtype=jnp.float32)
    k_nope = jax.random.normal(k3, (b, self.lkv_dim, s), dtype=jnp.float32)
    k_pe = jax.random.normal(k4, (b, self.r_dim, s), dtype=jnp.float32)

    m_prev = jnp.full((n_q, 128), -jnp.inf, dtype=jnp.float32)
    l_prev = jnp.zeros((n_q, 128), dtype=jnp.float32)
    o_prev = jnp.zeros((n_q, self.lkv_dim), dtype=jnp.float32)

    # 1. Chunked execution (q_split = 4, chunk size = 2)
    m_chunk, l_chunk, o_chunk = flash_attention.chunked_flash_attention(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        m_prev,
        l_prev,
        o_prev,
        cfgs=mla_cfg,
    )
    out_chunked = _normalize_output(
        o_chunk, l_chunk, out_dtype=jnp.float32
    )

    # 2. Unchunked single pass reference
    p_ref, a_ref, m_ref, l_ref = flash_attention.flash_attention_qk_softmax(
        q_nope, q_pe, k_nope, k_pe, m_prev, l_prev, cfgs=mla_cfg
    )
    acc_ref = flash_attention.flash_attention_pv(
        p_ref, k_nope, a_ref, o_prev, cfgs=mla_cfg
    )
    out_ref = _normalize_output(
        acc_ref, l_ref, out_dtype=jnp.float32
    )

    np.testing.assert_allclose(out_chunked, out_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()


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

"""Unit tests for the MLA Metadata Scheduler runnable on CPU."""

import jax.numpy as jnp
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import schedule
from absl.testing import absltest
from absl.testing import parameterized


class MlaScheduleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_cfg = configs.MlaModelConfigs(
        num_q_heads=128,
        lkv_dim=512,
        r_dim=64,
        mask_value=-1e9,
    )

  def test_decode_lane_balancing_and_dma_structs(self):
    """Verifies decode task assignment, dma_q, dma_kv_cache, and dma_kv_new."""
    batch_size = 2
    bq_sz = 1
    bkv_sz = 128
    page_size = 128

    # 2 decode sequences: 1 query token each, 128 total context tokens (127 cached + 1 new)
    kv_lens = jnp.array([128, 128], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 2], dtype=jnp.int32)
    page_indices = jnp.array([10, 20], dtype=jnp.int32)
    distribution = jnp.array([2, 2, 2], dtype=jnp.int32)  # 2 decode sequences

    block_cfg = configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_sz,
        bkv_sz=bkv_sz,
        batch_size=batch_size,
        n_buffer=2,
    )
    serving_cfg = configs.ServingConfigs(
        num_seqs=2,
        page_size=page_size,
        total_q_tokens=2,
        num_page_indices=2,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    mla_cfg = configs.MlaConfigs(
        block=block_cfg,
        model=self.model_cfg,
        serve=serving_cfg,
        mode=configs.MlaCase.DECODE,
        vmem_limit_bytes=16 * 1024 * 1024,
    )

    sched = schedule.generate_mla_metadata(
        cu_q_lens, kv_lens, page_indices, distribution, mla_cfg, interpret=True
    )

    # 1. Total Steps & Coordinates
    self.assertEqual(int(sched.actual_steps[0]), 1)
    self.assertEqual(int(sched.s_idx[0, 0]), 0)
    self.assertEqual(int(sched.s_idx[0, 1]), 1)
    self.assertEqual(int(sched.q_idx[0, 0]), 0)
    self.assertEqual(int(sched.q_idx[0, 1]), 0)
    self.assertEqual(int(sched.k_idx[0, 0]), 0)
    self.assertEqual(int(sched.k_idx[0, 1]), 0)
    self.assertEqual(int(sched.is_last_k[0, 0]), 1)
    self.assertEqual(int(sched.is_last_k[0, 1]), 1)
    self.assertEqual(int(sched.do_writeback[0, 0]), 1)
    self.assertEqual(int(sched.do_writeback[0, 1]), 1)

    # 2. Query DMA (dma_q): [src_hbm, sz]
    self.assertEqual(int(sched.dma_q[0, 0, 0]), 0)
    self.assertEqual(int(sched.dma_q[0, 0, 1]), 1)
    self.assertEqual(int(sched.dma_q[0, 1, 0]), 1)
    self.assertEqual(int(sched.dma_q[0, 1, 1]), 1)

    # 3. Cached KV DMA (dma_kv_cache): [src_hbm_page_table_offset, dst_vmem, dma_valid]
    self.assertEqual(int(sched.dma_kv_cache[0, 0, 0, 0]), 0)
    self.assertEqual(int(sched.dma_kv_cache[0, 0, 0, 1]), 0)
    self.assertEqual(int(sched.dma_kv_cache[0, 0, 0, 2]), 1)
    self.assertEqual(int(sched.dma_kv_cache[0, 1, 0, 0]), 1)
    self.assertEqual(int(sched.dma_kv_cache[0, 1, 0, 1]), 0)
    self.assertEqual(int(sched.dma_kv_cache[0, 1, 0, 2]), 1)

    # 4. New KV Tokens DMA Struct (dma_kv_new): SeqAlongLane (5 fields)
    entry_s0 = sched.dma_kv_new[0, 0, 0]
    self.assertEqual(int(entry_s0.fetch_hbm), 0)
    self.assertEqual(int(entry_s0.fetch_vmem), 128)
    self.assertEqual(int(entry_s0.wb_hbm), 0)
    self.assertEqual(int(entry_s0.wb_vmem), 0)
    self.assertEqual(int(entry_s0.fetch_val), 1)
    self.assertEqual(int(entry_s0.wb_val), 1)

    entry_s1 = sched.dma_kv_new[0, 1, 0]
    self.assertEqual(int(entry_s1.fetch_hbm), 0)
    self.assertEqual(int(entry_s1.fetch_vmem), 128)
    self.assertEqual(int(entry_s1.wb_hbm), 1)
    self.assertEqual(int(entry_s1.wb_vmem), 0)
    self.assertEqual(int(entry_s1.fetch_val), 1)
    self.assertEqual(int(entry_s1.wb_val), 1)

  def test_prefill_dma_structs_and_writeback_flags(self):
    """Verifies chunked prompt prefill with writeback deduplication."""
    batch_size = 2
    bq_sz = 64
    bkv_sz = 64
    page_size = 64

    # 1 sequence with 128 prompt tokens (2 Q blocks, 2 K blocks)
    kv_lens = jnp.array([128], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 128], dtype=jnp.int32)
    page_indices = jnp.array([5, 6], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

    block_cfg = configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_sz,
        bkv_sz=bkv_sz,
        batch_size=batch_size,
        n_buffer=2,
    )
    serving_cfg = configs.ServingConfigs(
        num_seqs=1,
        page_size=page_size,
        total_q_tokens=128,
        num_page_indices=2,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    mla_cfg = configs.MlaConfigs(
        block=block_cfg,
        model=self.model_cfg,
        serve=serving_cfg,
        mode=configs.MlaCase.PREFILL,
        vmem_limit_bytes=16 * 1024 * 1024,
    )

    sched = schedule.generate_mla_metadata(
        cu_q_lens, kv_lens, page_indices, distribution, mla_cfg, interpret=True
    )

    self.assertEqual(int(sched.actual_steps[0]), 2)

    # Step 0, Lane 0: (Q0, K0)
    self.assertEqual(int(sched.dma_q[0, 0, 0]), 0)
    self.assertEqual(int(sched.dma_q[0, 0, 1]), 64)
    entry_s0_q0_k0 = sched.dma_kv_new[0, 0, 0]
    self.assertEqual(int(entry_s0_q0_k0.fetch_hbm), 0)
    self.assertEqual(int(entry_s0_q0_k0.wb_hbm), 0)
    self.assertEqual(int(sched.do_writeback[0, 0]), 1)

    # Step 0, Lane 1: (Q1, K0)
    self.assertEqual(int(sched.dma_q[0, 1, 0]), 64)
    self.assertEqual(int(sched.dma_q[0, 1, 1]), 64)
    entry_s0_q1_k0 = sched.dma_kv_new[0, 1, 0]
    self.assertEqual(int(entry_s0_q1_k0.fetch_hbm), 0)
    self.assertEqual(int(sched.do_writeback[0, 1]), 0)

    # Step 1, Lane 0: (Q1, K1)
    self.assertEqual(int(sched.dma_q[1, 0, 0]), 64)
    self.assertEqual(int(sched.dma_q[1, 0, 1]), 64)
    entry_s0_q1_k1 = sched.dma_kv_new[1, 0, 0]
    self.assertEqual(int(entry_s0_q1_k1.fetch_hbm), 64)
    self.assertEqual(int(entry_s0_q1_k1.wb_hbm), 1)
    self.assertEqual(int(sched.do_writeback[1, 0]), 1)
    self.assertEqual(int(sched.is_last_k[1, 0]), 1)

  def test_wait_synchronization_counters(self):
    """Verifies that total_wait counters aggregate DMA byte transfers into 512B lane units."""
    batch_size = 1
    bq_sz = 64
    bkv_sz = 64
    page_size = 64

    kv_lens = jnp.array([64], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 64], dtype=jnp.int32)
    page_indices = jnp.array([1], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

    block_cfg = configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_sz,
        bkv_sz=bkv_sz,
        batch_size=batch_size,
        n_buffer=2,
    )
    serving_cfg = configs.ServingConfigs(
        num_seqs=1,
        page_size=page_size,
        total_q_tokens=64,
        num_page_indices=1,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    mla_cfg = configs.MlaConfigs(
        block=block_cfg,
        model=self.model_cfg,
        serve=serving_cfg,
        mode=configs.MlaCase.PREFILL,
        vmem_limit_bytes=16 * 1024 * 1024,
    )

    sched = schedule.generate_mla_metadata(
        cu_q_lens, kv_lens, page_indices, distribution, mla_cfg, interpret=True
    )

    expected_q_wait = (64 * mla_cfg.q_bytes_per_token) // 512
    self.assertEqual(int(sched.total_wait_q_in[0]), expected_q_wait)

    expected_kv_in_wait = (64 * mla_cfg.kv_bytes_per_token) // 512
    self.assertEqual(int(sched.total_wait_kv_in[0]), expected_kv_in_wait)

    self.assertEqual(int(sched.total_wait_kv_out[0]), expected_kv_in_wait)

    expected_o_wait = (64 * mla_cfg.o_bytes_per_token) // 512
    self.assertEqual(int(sched.total_wait_o_out[0]), expected_o_wait)

if __name__ == "__main__":
  absltest.main()

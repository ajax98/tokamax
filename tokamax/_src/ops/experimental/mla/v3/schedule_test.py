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

    # 3. Cached KV DMA (dma_kv_cache): [src_hbm_page_table_offset, dma_valid].
    # `dst_vmem` used to sit between the two, but it is `page_idx <<
    # page_size_log2` -- a compile-time constant -- so it is recomputed at the
    # use site in `MlaSchedule.get_dma_kv_cache` rather than stored.
    self.assertEqual(int(sched.dma_kv_cache[0, 0, 0, 0]), 0)
    self.assertEqual(int(sched.dma_kv_cache[0, 0, 0, 1]), 1)
    self.assertEqual(int(sched.dma_kv_cache[0, 1, 0, 0]), 1)
    self.assertEqual(int(sched.dma_kv_cache[0, 1, 0, 1]), 1)

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

    # Only the KV totals exist: `total_wait_q_in` and `total_wait_o_out` had
    # no consumer and were removed rather than computed and ignored.
    expected_kv_in_wait = (64 * mla_cfg.kv_bytes_per_token) // 512
    self.assertEqual(int(sched.total_wait_kv_in[0]), expected_kv_in_wait)

    self.assertEqual(int(sched.total_wait_kv_out[0]), expected_kv_in_wait)

class ScheduleCapacityTest(parameterized.TestCase):
  """The HBM schedule must be large enough that no flush is ever dropped.

  `_write_schedule_to_hbm` skips a flush that would overrun the buffer, which
  turns an out-of-bounds DMA into a silently truncated attention. The guard
  cannot do better - Pallas DMA slice sizes are static Python ints, so a partial
  flush is not expressible - so the buffer has to be sized correctly up front,
  from `MlaConfigs.max_steps_needed`. These tests are that bound's evidence:
  that it really does dominate what the scheduler emits, and that the sizing
  actually consumes it.
  """

  def setUp(self):
    super().setUp()
    self.model_cfg = configs.MlaModelConfigs(
        num_q_heads=128,
        lkv_dim=512,
        r_dim=64,
        mask_value=-1e9,
    )

  def _configs(
      self,
      *,
      num_seqs,
      pages_per_seq,
      page_size,
      total_q_tokens,
      bq_sz,
      bkv_sz,
      batch_size,
      mode,
      smem_fraction=0.33,
  ):
    return configs.MlaConfigs(
        block=configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=bq_sz,
            bkv_sz=bkv_sz,
            batch_size=batch_size,
            n_buffer=2,
        ),
        model=self.model_cfg,
        serve=configs.ServingConfigs(
            num_seqs=num_seqs,
            page_size=page_size,
            total_q_tokens=total_q_tokens,
            num_page_indices=num_seqs * pages_per_seq,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.bfloat16,
            dtype_out=jnp.bfloat16,
            smem_fraction_limit_for_schedule_generation=smem_fraction,
        ),
        mode=mode,
        vmem_limit_bytes=16 * 1024 * 1024,
    )

  # `kv_lens` are ragged and deliberately unaligned to `page_size` / `bkv_sz`,
  # since a bound that only holds on round numbers is not a bound.
  @parameterized.named_parameters(
      dict(
          testcase_name="decode",
          q_lens=[1, 1, 1, 1],
          kv_lens=[17, 128, 250, 256],
          pages_per_seq=2,
          page_size=128,
          bq_sz=1,
          bkv_sz=128,
          batch_size=2,
          mode=configs.MlaCase.DECODE,
      ),
      dict(
          # The split that saturates the bound: every sequence is one token
          # long and its context fills the page table exactly, so no causal
          # masking and no partial block gives anything back. Slack here is
          # zero, which is what makes this case load-bearing - an off-by-one in
          # `max_steps_needed` shows up as a failure rather than as headroom.
          testcase_name="decode_at_the_bound",
          q_lens=[1] * 8,
          kv_lens=[512] * 8,
          pages_per_seq=4,
          page_size=128,
          bq_sz=1,
          bkv_sz=128,
          batch_size=1,
          mode=configs.MlaCase.DECODE,
      ),
      dict(
          testcase_name="prefill",
          q_lens=[64, 128, 200],
          kv_lens=[64, 128, 200],
          pages_per_seq=4,
          page_size=64,
          bq_sz=64,
          bkv_sz=64,
          batch_size=2,
          mode=configs.MlaCase.PREFILL,
      ),
      dict(
          # kv_len > q_len: chunked prefill, where the causal window starts part
          # way into the cache and the q-block count no longer tracks the
          # k-block count.
          testcase_name="chunked_prefill",
          q_lens=[64, 32],
          kv_lens=[200, 256],
          pages_per_seq=4,
          page_size=64,
          bq_sz=32,
          bkv_sz=64,
          batch_size=1,
          mode=configs.MlaCase.MIXED,
      ),
      dict(
          # One sequence carrying the whole token budget, which is the split
          # that maximises q-blocks per sequence rather than sequence count.
          testcase_name="single_long_sequence",
          q_lens=[0, 0, 384],
          kv_lens=[0, 0, 384],
          pages_per_seq=4,
          page_size=128,
          bq_sz=32,
          bkv_sz=128,
          batch_size=4,
          mode=configs.MlaCase.PREFILL,
      ),
  )
  def test_bound_dominates_emitted_steps(
      self,
      q_lens,
      kv_lens,
      pages_per_seq,
      page_size,
      bq_sz,
      bkv_sz,
      batch_size,
      mode,
  ):
    num_seqs = len(q_lens)
    cfgs = self._configs(
        num_seqs=num_seqs,
        pages_per_seq=pages_per_seq,
        page_size=page_size,
        total_q_tokens=sum(q_lens),
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        batch_size=batch_size,
        mode=mode,
    )
    cu_q_lens = [0]
    for q_len in q_lens:
      cu_q_lens.append(cu_q_lens[-1] + q_len)

    match mode:
      case configs.MlaCase.DECODE:
        distribution = [num_seqs, num_seqs, num_seqs]
      case configs.MlaCase.PREFILL:
        distribution = [0, num_seqs, num_seqs]
      case configs.MlaCase.MIXED:
        distribution = [0, 0, num_seqs]

    sched = schedule.generate_mla_metadata(
        jnp.array(cu_q_lens, dtype=jnp.int32),
        jnp.array(kv_lens, dtype=jnp.int32),
        jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32),
        jnp.array(distribution, dtype=jnp.int32),
        cfgs,
        interpret=True,
    )

    self.assertLessEqual(int(sched.actual_steps[0]), cfgs.max_steps_needed)
    # `actual_steps` is clamped to the buffer capacity, so it would satisfy the
    # assertion above for free if the buffer were undersized. Check separately
    # that the capacity is not what made it fit.
    self.assertLessEqual(
        cfgs.max_steps_needed,
        cfgs.max_steps_ub * cfgs.max_schedule_size_multiplier,
    )

  def test_multiplier_is_raised_when_the_shape_needs_it(self):
    """A shape too big for the configured multiplier grows it, not truncates."""
    # A zero SMEM budget pins `max_steps_ub` to its floor of one lane group,
    # which makes the arithmetic here independent of the host's SMEM size.
    cfgs = self._configs(
        num_seqs=128,
        pages_per_seq=32,
        page_size=256,
        total_q_tokens=128,
        bq_sz=1,
        bkv_sz=512,
        batch_size=2,
        mode=configs.MlaCase.DECODE,
        smem_fraction=0.0,
    )

    # 128 one-token sequences x cdiv(8192, 512) k-blocks / 2 lanes.
    self.assertEqual(cfgs.max_steps_needed, 1024)
    self.assertGreater(cfgs.max_steps_needed, cfgs.max_steps_ub)
    self.assertGreater(cfgs.max_schedule_size_multiplier, 1)
    self.assertGreaterEqual(
        cfgs.max_steps_ub * cfgs.max_schedule_size_multiplier,
        cfgs.max_steps_needed,
    )

  def test_multiplier_is_one_when_the_shape_already_fits(self):
    """No headroom is bought when a single lane group is enough.

    `max_schedule_size_multiplier` used to be a configurable floor (default
    16) that sizing could raise but not lower. The floor never bound in
    practice -- a shape that needs headroom raises the multiplier to ~163
    against a floor of 16 -- so it is now purely derived, and a shape that
    fits gets exactly 1.
    """
    cfgs = self._configs(
        num_seqs=3,
        pages_per_seq=32,
        page_size=256,
        total_q_tokens=3,
        bq_sz=1,
        bkv_sz=512,
        batch_size=2,
        mode=configs.MlaCase.DECODE,
    )
    self.assertLess(cfgs.max_steps_needed, cfgs.max_steps_ub)
    self.assertEqual(cfgs.max_schedule_size_multiplier, 1)


if __name__ == "__main__":
  absltest.main()

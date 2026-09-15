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
# ==============================================================================
"""Pallas BufferedRef overrides for Batched MLA DMA pipelines.

  - KVBufferedRefSeqAlongLane: Handles transposed [C_kv, K_pe] cache with dual new KV inputs.
  - BatchingQNopeRef: Non-positional query (d_nope = 512).
  - BatchingQPeRef: Decoupled RoPE query (d_pe = 64).
  - BatchingORef: Output activations (d_nope = 512).
"""

import dataclasses
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.mla.v3 import configs
from tokamax._src.ops.experimental.mla.v3 import schedule


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _BypassRef(pltpu.BufferedRef):
  """Helper class to safely bypass buffer_count checks during creation."""

  def __post_init__(self):
    # Pallas restricts n_buffer > 2 for output refs by default; override to allow
    # flexible pipelined buffer depths.
    pass


def _rebuild_with_cfgs(cls, standard_ref, cfgs):
  """Re-wraps a plain `pltpu.BufferedRef` as `cls`, attaching `cfgs`.

  Every subclass in this module adds exactly one field (`cfgs`) to
  `pltpu.BufferedRef`, so construction is a field-by-field copy of the standard
  ref plus that field. The copy is keyed by field *name*, so a Pallas release
  that renames or drops a `BufferedRef` field raises `TypeError` here rather
  than silently mis-wiring a buffer.

  This reaches into Pallas internals: `pltpu.BufferedRef` is not a stable public
  API. Verified against jax 0.11.x; revisit on JAX upgrades.
  """
  return cls(
      cfgs=cfgs,
      **{
          f.name: getattr(standard_ref, f.name)
          for f in dataclasses.fields(pltpu.BufferedRef)
      },
  )


# ==============================================================================
# Transposed KV Cache BufferedRef (SEQ_ALONG_LANE)
# ==============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class KVBufferedRefSeqAlongLane(_BypassRef):
  """Handles fetching/updating KV cache using SEQ_ALONG_LANE memory layout."""

  cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pytype: disable=signature-mismatch
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: configs.MlaConfigs,
      **kwargs,
  ) -> "KVBufferedRefSeqAlongLane":
    assert buffer_type == pltpu.BufferType.INPUT_OUTPUT

    standard_ref = _BypassRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
        **kwargs,
    )
    return _rebuild_with_cfgs(cls, standard_ref, cfgs)

  def copy_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, jax.Ref, schedule.MlaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # src_ref: (kv_cache_hbm, new_kv_c_hbm, new_k_pe_hbm, schedule_ref, page_indices_ref)
    (
        kv_cache_hbm,
        new_kv_c_hbm,
        new_k_pe_hbm,
        schedule_ref,
        page_indices_ref,
    ) = src_ref

    slot = self.current_copy_in_slot
    assert self.sem_recvs is not None
    sem: Any = self.sem_recvs.at[slot]
    block_idx = jnp.maximum(grid_indices[0], 0)

    assert self.window_ref is not None
    vmem_dst_lane: Any = self.window_ref.at[slot]
    num_lanes = pltpu.get_tpu_info().num_lanes
    lkv_sublanes = self.cfgs.lkv_sublanes

    for b in range(self.cfgs.batch_size):
      # 1. Fetch cached paged tokens from 4D HBM cache
      for i in range(self.cfgs.bkv_p_cache):
        p_idx, dst_off, dma_valid = schedule_ref.get_dma_kv_cache(
            block_idx, b, i
        )
        hbm_p_idx = page_indices_ref[p_idx]
        sz = dma_valid * self.cfgs.serve.page_size
        dst_off = pl.multiple_of(dst_off, num_lanes)
        sz = pl.multiple_of(sz, num_lanes)

        # C_kv [:lkv_sublanes] and K_pe [lkv_sublanes : lkv+r] are adjacent
        # and share source and destination lane slices, so one copy is
        # identical to the two below and halves the descriptor count. Byte
        # totals are unchanged, so `total_wait_kv_in` still matches.
        pltpu.make_async_copy(
            kv_cache_hbm.at[hbm_p_idx, :, :, pl.ds(0, sz)],
            vmem_dst_lane.at[b, :, :, pl.ds(dst_off, sz)],
            sem,
        ).start()
      # 2. Fetch unpaged new KV tokens from HBM
      for i in range(self.cfgs.bkv_p_new):
        dma_entry = schedule_ref.dma_kv_new[block_idx, b, i]
        src_new_off = dma_entry.fetch_hbm[...]
        dst_vmem_off = dma_entry.fetch_vmem[...]
        dma_valid = dma_entry.fetch_val
        sz = dma_valid * self.cfgs.serve.page_size
        src_new_off = pl.multiple_of(src_new_off, num_lanes)
        dst_vmem_off = pl.multiple_of(dst_vmem_off, num_lanes)
        sz = pl.multiple_of(sz, num_lanes)

        def _start_new_kv(
            src_new_off=src_new_off, dst_vmem_off=dst_vmem_off, sz=sz, b=b
        ):
          # new_kv_c_hbm -> vmem_dst_lane[:lkv_sublanes, :, :] (C_kv)
          pltpu.make_async_copy(
              new_kv_c_hbm.at[:, :, pl.ds(src_new_off, sz)],
              vmem_dst_lane.at[b, :lkv_sublanes, :, pl.ds(dst_vmem_off, sz)],
              sem,
          ).start()

          # new_k_pe_hbm -> vmem_dst_lane[lkv_sublanes:, :, :] (K_pe).
          pltpu.make_async_copy(
              new_k_pe_hbm.at[:, :, pl.ds(src_new_off, sz)],
              vmem_dst_lane.at[
                  b, lkv_sublanes:, :, pl.ds(dst_vmem_off, sz)
              ],
              sem,
          ).start()

        _start_new_kv()

  def copy_out(
      self,
      dst_ref: tuple[jax.Ref, jax.Ref, jax.Ref, schedule.MlaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # dst_ref: (kv_out_ref, _, _, schedule_ref, page_indices_ref)
    kv_out_ref, _, _, schedule_ref, page_indices_ref = dst_ref
    slot = self.current_copy_out_slot
    assert self.sem_sends is not None
    sem: Any = self.sem_sends.at[slot]
    block_idx = grid_indices[0]

    assert self.window_ref is not None
    vmem_src_lane: Any = self.window_ref.at[slot]
    num_lanes = pltpu.get_tpu_info().num_lanes
    lkv_sublanes = self.cfgs.lkv_sublanes

    for b in range(self.cfgs.batch_size):
      do_writeback = schedule_ref.do_writeback[block_idx, b] == 1
      for i in range(self.cfgs.bkv_p_new):
        dma_entry = schedule_ref.dma_kv_new[block_idx, b, i]
        dst_hbm_p = dma_entry.wb_hbm[...]
        src_vmem_off = dma_entry.wb_vmem[...]
        dma_valid = dma_entry.wb_val
        hbm_p_idx = page_indices_ref[dst_hbm_p]
        sz = jnp.where(do_writeback, dma_valid * self.cfgs.serve.page_size, 0)
        src_vmem_off = pl.multiple_of(src_vmem_off, num_lanes)
        sz = pl.multiple_of(sz, num_lanes)

        # Write back concatenated into 4D kv_out_ref. Same adjacency argument
        # as `copy_in`: one copy covers both halves.
        def _start_wb(
            hbm_p_idx=hbm_p_idx, src_vmem_off=src_vmem_off, sz=sz, b=b
        ):
          pltpu.make_async_copy(
              vmem_src_lane.at[b, :, :, pl.ds(src_vmem_off, sz)],
              kv_out_ref.at[hbm_p_idx, :, :, pl.ds(0, sz)],
              sem,
          ).start()
        # Writeback only happens on the block that took in new tokens, so in
        # DECODE two thirds of these are empty. `total_wait_kv_out` is a byte
        # count, so skipping the issue does not perturb it.
        _start_wb()

  def wait_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, jax.Ref, schedule.MlaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    _, _, _, schedule_ref, _ = src_ref
    slot = self.current_wait_in_slot
    assert self.sem_recvs is not None
    sem: Any = self.sem_recvs.at[slot]
    block_idx = grid_indices[0]
    wait_lanes = schedule_ref.total_wait_kv_in[block_idx]

    assert self.window_ref is not None
    vmem_dst: Any = self.window_ref.at[slot]
    vmem_u32 = vmem_dst.bitcast(jnp.uint32)
    flat_dst = vmem_u32.reshape((-1, 128))
    pltpu.make_async_copy(
        flat_dst.at[pl.ds(0, wait_lanes), :],
        flat_dst.at[pl.ds(0, wait_lanes), :],
        sem,
    ).wait()

  def wait_out(
      self,
      dst_ref: tuple[jax.Ref, jax.Ref, jax.Ref, schedule.MlaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    _, _, _, schedule_ref, _ = dst_ref
    slot = self.current_wait_out_slot
    assert self.sem_sends is not None
    sem: Any = self.sem_sends.at[slot]
    block_idx = grid_indices[0]
    wait_lanes = schedule_ref.total_wait_kv_out[block_idx]

    assert self.window_ref is not None
    vmem_src: Any = self.window_ref.at[slot]
    vmem_u32 = vmem_src.bitcast(jnp.uint32)
    flat_src = vmem_u32.reshape((-1, 128))
    pltpu.make_async_copy(
        flat_src.at[pl.ds(0, wait_lanes), :],
        flat_src.at[pl.ds(0, wait_lanes), :],
        sem,
    ).wait()


# ==============================================================================
# Dedicated Query BufferedRefs (BatchingQNopeRef & BatchingQPeRef)
# ==============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingQNopeRef(pltpu.BufferedRef):
  """Handles fetching non-positional Query blocks (Q_nope, width d_nope = 512)."""

  cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pytype: disable=signature-mismatch
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: configs.MlaConfigs,
      **kwargs,
  ) -> "BatchingQNopeRef":
    assert buffer_type == pltpu.BufferType.INPUT

    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
        **kwargs,
    )
    return _rebuild_with_cfgs(cls, standard_ref, cfgs)

  def copy_in(
      self,
      src_ref: tuple[jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # src_ref: (q_nope_hbm, schedule_ref)
    q_nope_hbm, schedule_ref = src_ref
    slot = self.current_copy_in_slot
    assert self.sem_recvs is not None
    sem: Any = self.sem_recvs.at[slot]
    assert self.window_ref is not None
    vmem_dst: Any = self.window_ref.at[slot]
    block_idx = grid_indices[0]

    for b in range(self.cfgs.batch_size):
      q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
      pltpu.make_async_copy(
          q_nope_hbm.at[pl.ds(q_src, q_sz), ...],
          vmem_dst.at[b, pl.ds(0, q_sz), ...],
          sem,
      ).start()

  def wait_in(
      self,
      src_ref: tuple[jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    _, schedule_ref = src_ref
    slot = self.current_wait_in_slot
    assert self.sem_recvs is not None
    sem: Any = self.sem_recvs.at[slot]
    block_idx = grid_indices[0]

    q_in_tokens = 0
    for b in range(self.cfgs.batch_size):
      _, q_sz = schedule_ref.get_dma_q(block_idx, b)
      q_in_tokens += q_sz

    itemsize = jnp.dtype(self.cfgs.serve.dtype_q).itemsize
    # The minormost dimension of the u32 view, which the reshape below must
    # preserve. TPU packs sub-word types along *sublanes*, so `bitcast` divides
    # the second-minor axis by the packing and leaves the lane axis alone: the
    # packed 4D buffer ends ...x packing x 128 and views as minor 128, whereas
    # `flat_q`'s 3D buffer ends x aligned_lkv_dim and views as minor 512.
    # Reshaping that to (-1, 128) is what Mosaic rejects with "Expected the
    # minormost dimension to be unchanged".
    minor = self.cfgs.aligned_lkv_dim
    dma_chunk_size = minor * 4
    q_nope_bytes_per_token = (
        self.cfgs.aligned_num_q_heads * self.cfgs.aligned_lkv_dim * itemsize
    )
    # Fold the division into a Python constant before multiplying by the
    # dynamic token count. Written as `(q_in_tokens * bytes) // chunk` the
    # result is a dynamic floor-divide, and Mosaic cannot prove the slice below
    # is sublane-aligned ("size at dimension 0 divisible by the tile dimension
    # 8"). As `q_in_tokens * const` the multiple is syntactically visible.
    rows_per_token = q_nope_bytes_per_token // dma_chunk_size
    assert rows_per_token % 8 == 0, (
        f"{rows_per_token=} must be sublane-aligned for the wait slice"
    )
    wait_lanes = q_in_tokens * rows_per_token

    assert self.window_ref is not None
    vmem_dst: Any = self.window_ref.at[slot]
    vmem_u32 = vmem_dst.bitcast(jnp.uint32)
    flat_vmem = vmem_u32.reshape((-1, minor))
    pltpu.make_async_copy(
        flat_vmem.at[pl.ds(0, wait_lanes), :],
        flat_vmem.at[pl.ds(0, wait_lanes), :],
        sem,
    ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingQPeRef(pltpu.BufferedRef):
  """Handles fetching decoupled RoPE Query blocks (Q_pe, width d_pe = 64)."""

  cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pytype: disable=signature-mismatch
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: configs.MlaConfigs,
      **kwargs,
  ) -> "BatchingQPeRef":
    assert buffer_type == pltpu.BufferType.INPUT

    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
        **kwargs,
    )
    return _rebuild_with_cfgs(cls, standard_ref, cfgs)

  def copy_in(
      self,
      src_ref: tuple[jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # src_ref: (q_pe_hbm, schedule_ref)
    q_pe_hbm, schedule_ref = src_ref
    slot = self.current_copy_in_slot
    assert self.sem_recvs is not None
    sem: Any = self.sem_recvs.at[slot]
    assert self.window_ref is not None
    vmem_dst: Any = self.window_ref.at[slot]
    block_idx = grid_indices[0]

    for b in range(self.cfgs.batch_size):
      q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
      pltpu.make_async_copy(
          q_pe_hbm.at[pl.ds(q_src, q_sz), ...],
          vmem_dst.at[b, pl.ds(0, q_sz), ...],
          sem,
      ).start()

  def wait_in(
      self,
      src_ref: tuple[jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    _, schedule_ref = src_ref
    slot = self.current_wait_in_slot
    assert self.sem_recvs is not None
    sem: Any = self.sem_recvs.at[slot]
    block_idx = grid_indices[0]

    q_in_tokens = 0
    for b in range(self.cfgs.batch_size):
      _, q_sz = schedule_ref.get_dma_q(block_idx, b)
      q_in_tokens += q_sz

    itemsize = jnp.dtype(self.cfgs.serve.dtype_q).itemsize
    dma_chunk_size = 128 * 4
    q_pe_bytes_per_token = (
        self.cfgs.aligned_num_q_heads * self.cfgs.aligned_r_dim * itemsize
    )
    # Constant-folded for the same reason as in `BatchingQNopeRef.wait_in`.
    rows_per_token = q_pe_bytes_per_token // dma_chunk_size
    wait_lanes = q_in_tokens * rows_per_token

    assert self.window_ref is not None
    vmem_dst: Any = self.window_ref.at[slot]
    vmem_u32 = vmem_dst.bitcast(jnp.uint32)
    flat_vmem = vmem_u32.reshape((-1, 128))
    pltpu.make_async_copy(
        flat_vmem.at[pl.ds(0, wait_lanes), :],
        flat_vmem.at[pl.ds(0, wait_lanes), :],
        sem,
    ).wait()


# ==============================================================================
# Output Activation BufferedRef (BatchingORef)
# ==============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingORef(pltpu.BufferedRef):
  """Handles storing final attention output activations (width d_nope = 512)."""

  cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pytype: disable=signature-mismatch
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: configs.MlaConfigs,
      **kwargs,
  ) -> "BatchingORef":
    assert buffer_type == pltpu.BufferType.OUTPUT

    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
        **kwargs,
    )
    return _rebuild_with_cfgs(cls, standard_ref, cfgs)

  def copy_out(
      self,
      dst_ref: tuple[jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # dst_ref: (o_hbm, schedule_ref)
    o_hbm, schedule_ref = dst_ref
    slot = self.current_copy_out_slot
    assert self.sem_sends is not None
    sem: Any = self.sem_sends.at[slot]
    assert self.window_ref is not None
    vmem_src: Any = self.window_ref.at[slot]
    block_idx = grid_indices[0]

    for b in range(self.cfgs.batch_size):
      is_last_k = schedule_ref.is_last_k[block_idx, b] == 1
      q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
      q_sz = jnp.where(is_last_k, q_sz, 0)

      def _start_o(q_src=q_src, q_sz=q_sz, b=b):
        pltpu.make_async_copy(
            vmem_src.at[b, pl.ds(0, q_sz), ...],
            o_hbm.at[pl.ds(q_src, q_sz), ...],
            sem,
        ).start()

      # Output is emitted only at a sequence's last k-block; the rest issue a
      # zero-length copy. `wait_out` counts bytes, so this is invisible to it.
      _start_o()

  def wait_out(
      self,
      dst_ref: tuple[jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # dst_ref: (o_hbm, schedule_ref)
    _, schedule_ref = dst_ref
    slot = self.current_wait_out_slot
    assert self.sem_sends is not None
    sem: Any = self.sem_sends.at[slot]
    block_idx = grid_indices[0]
    # `_compute_waits` emits this in rows of the same `minor`-wide u32 view
    # the slice below uses, so it needs no rescaling here. Mosaic cannot see
    # that a value read out of SMEM is sublane-aligned, hence the hint; the
    # schedule asserts the property where the operands are Python ints.
    minor = self.cfgs.aligned_lkv_dim
    wait_lanes = pl.multiple_of(schedule_ref.total_wait_o_out[block_idx], 8)
    assert self.window_ref is not None
    vmem_src: Any = self.window_ref.at[slot]
    vmem_u32 = vmem_src.bitcast(jnp.uint32)
    flat_src = vmem_u32.reshape((-1, minor))
    pltpu.make_async_copy(
        flat_src.at[pl.ds(0, wait_lanes), :],
        flat_src.at[pl.ds(0, wait_lanes), :],
        sem,
    ).wait()



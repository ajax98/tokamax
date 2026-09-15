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
"""Tokamax `Op` wrapper for the experimental MLA v2 kernel.

Exists so that `v2/kernel.py` can be reached by `tokamax.autotune` and by
`tokamax/benchmarks/mla.py`, both of which dispatch through
`mla.api.IMPLEMENTATIONS` and therefore only see registered `Op`s.

See `v3_op.py` for the v3 counterpart, and note that the two `Config`s are
deliberately *not* symmetric -- `decode_batch_size` here and `batch_size` there
are unrelated parameters. See the "batch_size semantics trap" discussion: in v2
a lane is an independent sequence with its own `(m, l, acc)`; in v3 a lane is a
consecutive schedule task and the lanes share one `(m, l, acc)` chained
serially.
"""

import itertools
from typing import Annotated, ClassVar, override

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import pydantic
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.mla import base
from tokamax._src.ops.experimental.mla.v2 import kernel as kernel_v2


@pydantic.dataclasses.dataclass(frozen=True)
class Config:
  """Tuning configuration for the MLA v2 kernel.

  `num_kv_pages_per_block` and `num_queries_per_block` are scalars here, though
  the kernel also accepts a (decode, prefill, mixed) triple. A scalar is applied
  to all three cases, matching how `pallas_mosaic_tpu.Config` handles the same
  parameters and keeping the autotuning space tractable.

  Attributes:
    num_kv_pages_per_block: KV pages per flash-attention block.
    num_queries_per_block: Query tokens per flash-attention block.
    vmem_limit_bytes: VMEM budget passed to the Mosaic compiler.
    decode_batch_size: Number of *independent sequences* packed into one
      batched-decode grid step. Must divide the decode band; the kernel runs a
      second `batch_size=1` pass over the remainder.
    mixed_q_split: Query sub-chunking in the MIXED pass. Must divide
      `num_queries_per_block` or the kernel raises.
    chunk_prefill_size: Static query length for the PREFILL pass. Zero means
      `None`, which *skips the PREFILL pass entirely* -- sequences in
      `[distribution[0], distribution[1])` are then dropped. Follows the
      zero-means-None convention from `pallas_mosaic_tpu.Config`.
    two_step_flash_attention: Split QK+softmax and PV into two steps so the
      previous chunk's PV overlaps the next chunk's QK.
    p_same_dtype_as_v: Cast the softmax probabilities to the KV dtype before the
      PV matmul. This is v2's single largest precision lever and has no v3
      equivalent; at FP8 it is a much bigger MXU win than at bf16.
  """

  num_kv_pages_per_block: Annotated[int, pydantic.Field(gt=0)]
  num_queries_per_block: Annotated[int, pydantic.Field(gt=0)]
  vmem_limit_bytes: Annotated[int, pydantic.Field(multiple_of=16, gt=0)]
  decode_batch_size: Annotated[int, pydantic.Field(gt=0)]
  mixed_q_split: Annotated[int, pydantic.Field(gt=0)] = 1
  chunk_prefill_size: Annotated[int, pydantic.Field(ge=0)] = 0
  two_step_flash_attention: bool = True
  p_same_dtype_as_v: bool = True


class V2MultiHeadLatentAttention(base.MultiHeadLatentAttention):
  """Tokamax operator invoking the experimental MLA v2 kernel."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      ql_nope: Float[Array, "max_num_tokens actual_num_q_heads actual_lkv_dim"],
      q_pe: Float[Array, "max_num_tokens actual_num_q_heads actual_r_dim"],
      new_kv_c: Float[Array, "max_num_tokens actual_lkv_dim"],
      new_k_pe: Float[Array, "max_num_tokens actual_r_dim"],
      cache_kv: Float[
          Array,
          "total_num_pages page_size_per_kv_packing kv_packing lkv_dim",
      ],
      kv_lens: Int[Array, "max_num_seqs"],
      page_indices: Int[Array, "num_page_indices"],
      cu_q_lens: Int[Array, "max_num_seqs_plus_1"],
      distribution: Int[Array, "3"],
      *,
      sm_scale: float = 1.0,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
      mask_value: float | None = None,
      q_scale: float | None = None,
      k_scale: float | None = None,
      v_scale: float | None = None,
      s_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      debug_mode: bool = False,
      return_residuals: bool = False,
      config: Config | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    del return_residuals  # No residuals; forward-only op.
    assert config is not None, "Config must be specified."

    # The `Op` contract is token-major `[T, H, L]` (see `base.py` and
    # `reference.mla_attention`), but v2 takes and returns head-major
    # `[H, T, L]`. `q_pe` is already token-major in both, so only `ql_nope` and
    # the output are transposed -- the same asymmetry the v2 test applies.
    out, updated_cache_kv = kernel_v2.mla_ragged_paged_attention(
        jnp.transpose(ql_nope, (1, 0, 2)),
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=(
            config.chunk_prefill_size if config.chunk_prefill_size > 0 else None
        ),
        num_kv_pages_per_block=config.num_kv_pages_per_block,
        num_queries_per_block=config.num_queries_per_block,
        vmem_limit_bytes=config.vmem_limit_bytes,
        decode_batch_size=config.decode_batch_size,
        mixed_q_split=config.mixed_q_split,
        s_dtype=s_dtype,
        transpose_kv_cache=False,
        two_step_flash_attention=config.two_step_flash_attention,
        p_same_dtype_as_v=config.p_same_dtype_as_v,
        debug_mode=debug_mode,
    )
    return (jnp.transpose(out, (1, 0, 2)), updated_cache_kv), None

  @override
  def _get_heuristics_config(self, ba) -> Config:
    return Config(
        num_kv_pages_per_block=16,
        num_queries_per_block=1,
        vmem_limit_bytes=64 * 1024 * 1024,
        decode_batch_size=1,
        mixed_q_split=1,
        chunk_prefill_size=0,
    )

  @override
  def _get_autotuning_configs(self, ba) -> set[Config]:
    configs = set()
    for kv, q, decode_batch_size, vmem_mib in itertools.product(
        # 3 comes from the production DS-V3 decode sweep and is not a power of
        # two, so a naive power-of-two grid would miss it.
        [1, 3, 4, 8, 16, 32],
        [1, 4, 16],
        [1, 8],
        [48, 64],
    ):
      for mixed_q_split in [1, 16]:
        # The kernel raises unless `num_queries_per_block % mixed_q_split == 0`.
        if q % mixed_q_split != 0:
          continue
        configs.add(
            Config(
                num_kv_pages_per_block=kv,
                num_queries_per_block=q,
                vmem_limit_bytes=vmem_mib * 1024 * 1024,
                decode_batch_size=decode_batch_size,
                mixed_q_split=mixed_q_split,
                chunk_prefill_size=0,
            )
        )
    return configs

  @override
  def supported_on(self, device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5

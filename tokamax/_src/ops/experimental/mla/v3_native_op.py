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
"""MLA v3 `Op` that takes the KV cache already in v3's native layout.

Identical to `v3_op` except that it does **not** transpose the cache on the way
in or out. `v3_op` answers "what does dropping v3 in today cost?"; this answers
"what does it cost once the cache is migrated?".

Motivation, measured: on `decode_f8_kv9216` the per-call conversion is ~10.7 ms
of v3's 11.64 ms, against an attention kernel of 0.972 ms. A serving stack that
holds the cache in v3 layout permanently never pays it.

Requires an arg-spec built with `cache_layout='v3'` --
`[pages, aligned_kv_dim // P, P, page_size]`, e.g. `(4608, 160, 4, 256)` rather
than `(4608, 64, 4, 640)`. Both are 4-D float arrays so jaxtyping accepts
either, but `v3.kernel.static_validate_inputs` recomputes
`kv_dim = kv_sublanes * kv_packing` and rejects the v2 shape against
`lkv_dim + r_dim`. Loudly, which is what we want.

Note the updated cache is returned in v3 layout too, so its shape differs from
v2's output. That is fine for benchmarking; correctness of the underlying kernel
is covered by `v2_v3_op_test.py` through the converting `v3_op`.
"""

import itertools
from typing import ClassVar, override

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.mla import base
from tokamax._src.ops.experimental.mla import v3_op
from tokamax._src.ops.experimental.mla.v3 import mla_wrapper


# Same tuning surface as `v3_op`; reused rather than redeclared so the two
# cannot drift apart.
Config = v3_op.Config


class V3NativeMultiHeadLatentAttention(base.MultiHeadLatentAttention):
  """MLA v3 with a natively-laid-out KV cache (no per-call transpose)."""

  config_cls: ClassVar[type[Config]] = Config

  @override
  @jaxtyping.jaxtyped
  def _fwd(
      self,
      ql_nope: Float[Array, "max_num_tokens actual_num_q_heads actual_lkv_dim"],
      q_pe: Float[Array, "max_num_tokens actual_num_q_heads actual_r_dim"],
      new_kv_c: Float[Array, "max_num_tokens actual_lkv_dim"],
      new_k_pe: Float[Array, "max_num_tokens actual_r_dim"],
      # Named for the base contract; the actual layout here is
      # [pages, aligned_kv_dim // P, P, page_size].
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
    # v3 computes scores and probabilities in f32 unconditionally, so `s_dtype`
    # cannot be honoured. Accepted and dropped so one arg-spec drives both ops.
    del s_dtype, return_residuals

    assert config is not None, "Config must be specified."

    # The only difference from `v3_op`: no `transpose_kv_cache_to_v3` here and
    # no `transpose_kv_cache_from_v3` on the result.
    # `ql_nope` arrives token-major, `[total_q_tokens, heads, lkv_dim]`. The
    # wrapper's historical contract is head-major, so it was transposed here
    # only for `prepare_q_nope_inputs` to transpose it straight back -- and
    # symmetrically on the way out. Four full passes over an 8.4 MB array on
    # the layout it actually wants.
    out, updated_cache_kv = mla_wrapper.mla_ragged_paged_attention(
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
        num_kv_pages_per_block=config.num_kv_pages_per_block,
        num_queries_per_block=config.num_queries_per_block,
        batch_size=config.batch_size,
        n_buffer=config.n_buffer,
        vmem_limit_bytes=config.vmem_limit_bytes,
        s_dtype=None,
        p_same_dtype_as_v=config.p_same_dtype_as_v,
        kv_slack_pad_lanes=config.kv_slack_pad_lanes,
        q_split=config.q_split,
        debug_mode=debug_mode,
    )
    out = jnp.transpose(out, (1, 0, 2))
    return (out, updated_cache_kv), None

  @override
  def _get_heuristics_config(self, ba) -> Config:
    return v3_op.V3MultiHeadLatentAttention()._get_heuristics_config(ba)  # pylint: disable=protected-access

  @override
  def _get_autotuning_configs(self, ba) -> set[Config]:
    configs = set()
    for kv, q, batch_size, n_buffer in itertools.product(
        [1, 2, 3, 4, 8, 16],
        # 1 included to match v2's [1, 4, 16]. On a decode-only distribution
        # this axis is inert -- v3 pins bq_sz=1 in `calculate_block_sizes` and
        # v2 clamps via `min(num_queries_per_block, static_q_len=1)` -- but it
        # is live for prefill/mixed, where omitting 1 gave v2 a config v3 could
        # not reach.
        [1, 4, 16],
        [1, 2, 4],
        [2, 3],
    ):
      configs.add(
          Config(
              num_kv_pages_per_block=kv,
              num_queries_per_block=q,
              batch_size=batch_size,
              n_buffer=n_buffer,
              vmem_limit_bytes=64 * 1024 * 1024,
          )
      )
    return configs

  @override
  def supported_on(self, device) -> bool:
    return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 5

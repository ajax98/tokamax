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
"""Multi-Head Latent Attention benchmark argument specifications."""

from collections.abc import Sequence
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.autotuning import arg_spec
from tokamax._src.ops.experimental.mla import utils


class HashableNPArray(np.ndarray):

  def __new__(cls, input_array):
    return np.asarray(input_array).view(cls)

  def __hash__(self):
    return hash((self.tobytes(), self.shape, self.dtype))


def _make_mla_spec(
    name: str,
    seq_lens: Sequence[tuple[int, int]],
    num_heads: int,
    lkv_dim: int,
    r_dim: int,
    page_size: int,
    q_dtype: jax.typing.DTypeLike,
    kv_dtype: jax.typing.DTypeLike,
    num_pages: int,
    tags: Sequence[arg_spec.Tag] = ('primary', 'forward_only', 'ci_tests'),
    cache_layout: str = 'v2',
) -> arg_spec.ArgSpec:
  """Generates an argument specification for MLA."""
  padded_r_dim = utils.align_to(r_dim, 128)
  padded_lkv_dim = utils.align_to(lkv_dim, 128)
  padded_kv_dim = padded_lkv_dim + padded_r_dim
  packing = utils.get_dtype_packing(kv_dtype)
  q_lens = [s[0] for s in seq_lens]
  kv_lens_list = [s[1] for s in seq_lens]
  total_q_len = sum(q_lens)
  cu_q_lens_list = [0]
  for q_len in q_lens:
    cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)

  max_kv_len = max(kv_lens_list) if kv_lens_list else 0
  pages_per_seq = utils.cdiv(max_kv_len, page_size)

  page_indices_list = []
  page_count = 0
  for kv_len in kv_lens_list:
    num_seq_pages = utils.cdiv(kv_len, page_size)
    indices = list(range(page_count, page_count + num_seq_pages))
    page_indices_list.extend(indices + [-1] * (pages_per_seq - num_seq_pages))
    page_count += num_seq_pages

  total_num_pages = max(num_pages, page_count)

  num_decode_seqs = 0
  for s in seq_lens:
    if s[0] == 1:
      num_decode_seqs += 1
    else:
      break
  distribution_list = [num_decode_seqs, num_decode_seqs, len(seq_lens)]

  ql_nope = jax.ShapeDtypeStruct((total_q_len, num_heads, lkv_dim), q_dtype)
  q_pe = jax.ShapeDtypeStruct((total_q_len, num_heads, r_dim), q_dtype)
  new_kv_c = jax.ShapeDtypeStruct((total_q_len, lkv_dim), kv_dtype)
  new_k_pe = jax.ShapeDtypeStruct((total_q_len, r_dim), kv_dtype)
  if cache_layout == 'v2':
    # Tokens on sublanes: [pages, page_size // P, P, kv_dim].
    cache_shape = (total_num_pages, page_size // packing, packing, padded_kv_dim)
  elif cache_layout == 'v3_compact':
    # As 'v3', but without the wasted lane-alignment padding. v3 stores kv_dim
    # on sublanes, so each part only needs a whole sublane group: 512 + 64 =
    # 576 rather than align_to(512,128) + align_to(64,128) = 640. At fp8 that
    # is 144 sublanes (a multiple of 8), so it is a legal layout and 10%
    # smaller. The kernel reads the width off the cache; no flag needed.
    compact_kv_dim = utils.align_to(lkv_dim, packing * 8) + utils.align_to(
        r_dim, packing * 8
    )
    cache_shape = (total_num_pages, compact_kv_dim // packing, packing,
                   page_size)
  elif cache_layout == 'v3':
    # SEQ_ALONG_LANE -- tokens on lanes, latent dim on sublanes:
    # [pages, aligned_kv_dim // P, P, page_size]. Same bytes, transposed.
    # A v3-layout cache is what `v3_native` consumes; feeding it the v2 shape
    # would make `static_validate_inputs` compute kv_dim = 64*4 = 256 and reject
    # it against lkv_dim + r_dim = 640.
    cache_shape = (total_num_pages, padded_kv_dim // packing, packing, page_size)
  else:
    raise ValueError(f'unknown {cache_layout=}')
  cache_kv = jax.ShapeDtypeStruct(cache_shape, kv_dtype)

  kv_lens = HashableNPArray(np.array(kv_lens_list, dtype=np.int32))
  page_indices = HashableNPArray(np.array(page_indices_list, dtype=np.int32))
  cu_q_lens = HashableNPArray(np.array(cu_q_lens_list, dtype=np.int32))
  distribution = HashableNPArray(np.array(distribution_list, dtype=np.int32))

  return arg_spec.ArgSpec(
      args=dict(
          ql_nope=ql_nope,
          q_pe=q_pe,
          new_kv_c=new_kv_c,
          new_k_pe=new_k_pe,
          cache_kv=cache_kv,
          kv_lens=kv_lens,
          page_indices=page_indices,
          cu_q_lens=cu_q_lens,
          distribution=distribution,
      ),
      project='deepseek_v3',
      name=name,
      tags=tuple(tags),
  )


ARG_SPECS: Final[tuple[arg_spec.ArgSpec, ...]] = (
    _make_mla_spec(
        name='decode_bf16',
        seq_lens=[(1, 8192)] * 3,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        num_pages=128,
    ),
    _make_mla_spec(
        # v3-layout twin of `decode_bf16`. Only 3 sequences, so this is the
        # smallest workload in the set and the one where fixed per-call
        # overhead matters most relative to the kernel.
        name='decode_bf16_v3layout',
        seq_lens=[(1, 8192)] * 3,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        num_pages=128,
        cache_layout='v3',
    ),
    _make_mla_spec(
        name='prefill_bf16',
        seq_lens=[(2048, 2048)] * 2,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        num_pages=128,
    ),
    _make_mla_spec(
        # `prefill_bf16` with the cache already in v3's SEQ_ALONG_LANE layout,
        # so `v3_native` can run it. Without this the only v3 path for this
        # shape is `v3_op`, which transposes the cache in and out on every
        # call *inside* the measured region -- a cost v2 never pays, which
        # makes the comparison unfair to v3 rather than merely noisy.
        #
        # Whole-prompt prefill: q_len == kv_len, so there is no cached history
        # and `_make_mla_spec` derives distribution [0, 0, 2], routing the
        # whole thing through the MIXED band.
        name='prefill_bf16_v3layout',
        seq_lens=[(2048, 2048)] * 2,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        num_pages=128,
        cache_layout='v3',
    ),
    _make_mla_spec(
        # `prefill_bf16_v3layout` with the KV dimension padded to sublane
        # granularity rather than lane granularity: 576 rather than 640, for a
        # (pages, 288, 2, 256) cache; the width is read off the array.
        #
        # The motivation differs from the decode compact spec. There it was
        # about *bytes*, and it bought 0.16% because decode is not
        # bandwidth-bound. Prefill is compute-bound -- roughly 1.55 TFLOP in
        # 2.69 ms, ~576 TFLOP/s of MXU -- and `aligned_kv_dim` widens the QK
        # contraction from 576 to 640, so 10% of that dot is multiplying
        # padding. Here the saving is FLOPs, not DMA.
        name='prefill_bf16_v3compact',
        seq_lens=[(2048, 2048)] * 2,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        num_pages=128,
        cache_layout='v3_compact',
    ),
    _make_mla_spec(
        # `chunked_prefill_f8_kv1024` with the cache already in v3's
        # SEQ_ALONG_LANE layout, so `v3_native` can run it without the per-call
        # transpose that would otherwise be charged to v3 alone. Same rationale
        # as `prefill_bf16_v3layout`.
        #
        # Unlike `prefill_bf16` this shape has real cached history -- kv_len
        # 1024 against q_len 256, so 768 cached tokens and 256 new. That is
        # both the case that exposed the `bq_sz == 1` stitch bug and the one
        # where v3's causal block skipping has less to skip, since the cached
        # 768 are visible to every query token.
        name='chunked_prefill_f8_kv1024_v3layout',
        seq_lens=[(256, 1024)],
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
        cache_layout='v3',
    ),
    _make_mla_spec(
        # v3-layout twin of `decode_f8`. Same 128-sequence decode shape as
        # `decode_f8_kv9216` but kv_len 8192 at page_size 256, so 32 pages per
        # sequence rather than 9 at page 1024 -- four times the DMA descriptors
        # per unit of KV, which is the axis decode turned out to be sensitive
        # to.
        name='decode_f8_v3layout',
        seq_lens=[(1, 8192)] * 128,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
        cache_layout='v3',
    ),
    _make_mla_spec(
        name='decode_f8',
        seq_lens=[(1, 8192)] * 128,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
    ),
    # ------------------------------------------------------------------------
    # MLA v2-vs-v3 comparison specs.
    #
    # Derived from the production DeepSeek-V3 sweep at page_size=256. The sweep
    # lists six entries at that page size, but they collapse to these three
    # distinct workloads -- the other three differ only in
    # `num_queries_per_block`, which is a tuning `Config` value, not part of the
    # workload.
    #
    # FP8 throughout (queries, new KV updates and the paged cache), mirroring
    # DeepSeek-V3 / Kimi FP8 KV-cache inference on TPU. `kv_dtype` also sets
    # `packing = 4`, so `cache_kv` is `[pages, page_size // 4, 4, 640]`.
    #
    # Tagged `primary` (required -- `benchmarks/mla.py` filters on it) but not
    # `ci_tests`: these are heavy shapes, and `decode_f8_kv9216` alone allocates
    # ~755 MB of KV cache. `ci_tests` is inert for MLA in any case; only
    # ragged_dot, attention and normalization consume it.
    #
    # Known gap: `_make_mla_spec` derives `distribution` as `[n, n, total]`, so
    # the PREFILL band is empty by construction and the two prefill shapes below
    # route through MIXED. Closing that needs a change to the builder.
    # ------------------------------------------------------------------------
    _make_mla_spec(
        # Chunked prefill with 7936 tokens of history already in the paged
        # cache. The highest-value spec for v2-vs-v3: nothing in the original
        # set has `kv_len > q_len`, so v3's `bkv_p_cache` path -- whose
        # docstring records that PREFILL used to skip the cache fetch entirely
        # -- is otherwise never exercised. Longest KV walk here, so also the
        # most likely to push v3's schedule past `max_steps_ub`.
        name='chunked_prefill_f8_kv8192',
        seq_lens=[(256, 8192)],
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
        tags=('primary', 'forward_only'),
    ),
    _make_mla_spec(
        # v3-layout twin of the above, so `v3_native` can run it without the
        # per-call cache transpose that would be charged to v3 alone.
        #
        # This is the extreme of the cached-history axis: 7936 of 8192 KV
        # tokens are already paged, so the causal block skipping that produces
        # v3's whole-prompt prefill advantage has essentially nothing to skip.
        # Expected to be v3's worst prefill case and the useful counterweight
        # to `prefill_bf16`.
        name='chunked_prefill_f8_kv8192_v3layout',
        seq_lens=[(256, 8192)],
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
        tags=('primary', 'forward_only'),
        cache_layout='v3',
    ),
    _make_mla_spec(
        # Pure decode at production batch size. Overlaps `decode_f8` above, but
        # at kv_len=9216 -- exactly 36 pages -- so the single new token lands at
        # offset 255, the last lane of its page. Edge case for both kernels'
        # KV stitch logic.
        name='decode_f8_kv9216',
        seq_lens=[(1, 9216)] * 128,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
        tags=('primary', 'forward_only'),
    ),
    _make_mla_spec(
        # Same shape as `chunked_prefill_f8_kv8192` with 768 rather than 7936
        # tokens of history. Weak alone; the point is the contrast pair, which
        # isolates how cost scales with cached history at fixed `q_len`.
        name='chunked_prefill_f8_kv1024',
        seq_lens=[(256, 1024)],
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
        tags=('primary', 'forward_only'),
    ),
    # ---- page_size=1024 pair -------------------------------------------
    # Same workload as decode_f8_kv9216, at the page size the production DS-V3
    # sweep actually uses for decode.
    #
    # Motivation: v3's SEQ_ALONG_LANE cache has `page_size` as its *minor*
    # dimension, so one contiguous HBM run is one page of tokens for a single
    # (sublane, packing) channel -- 256 bytes at page_size=256, against v2's
    # 640-byte runs (v2's minor dim is kv_dim). Decode is bandwidth-bound
    # (v2 1.46 TB/s vs v3 0.96 TB/s moving identical bytes), so run length may
    # be the whole gap. At page_size=1024 v3's runs become 1024 B -- larger
    # than v2's 640 -- which should flip the advantage.
    #
    # Both caches are still 755 MB; only the run length differs.
    _make_mla_spec(
        name='decode_f8_kv9216_p1024',
        seq_lens=[(1, 9216)] * 128,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=1024,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
        tags=('primary', 'forward_only'),
    ),
    _make_mla_spec(
        # page_size=1024 with the compact 576-wide KV cache: (1152, 144, 4,
        # 1024) = 679.5 MB against 755.0 MB. Pairs with
        # the array.
        name='decode_f8_kv9216_p1024_v3compact',
        seq_lens=[(1, 9216)] * 128,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=1024,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
        tags=('primary', 'forward_only'),
        cache_layout='v3_compact',
    ),
    _make_mla_spec(
        name='decode_f8_kv9216_p1024_v3layout',
        seq_lens=[(1, 9216)] * 128,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=1024,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
        tags=('primary', 'forward_only'),
        cache_layout='v3',
    ),
    _make_mla_spec(
        # As `decode_f8_kv9216_v3layout`, but with the KV dimension padded to
        # sublane granularity instead of lane granularity: 576 rather than 640,
        # giving a (4608, 144, 4, 256) cache. 10% less KV in HBM and 10% less
        # DMA per block; the width is read off the array.
        name='decode_f8_kv9216_v3compact',
        seq_lens=[(1, 9216)] * 128,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
        tags=('primary', 'forward_only'),
        cache_layout='v3_compact',
    ),
    _make_mla_spec(
        # Byte-for-byte the same workload as `decode_f8_kv9216`, with the KV
        # cache already in v3's SEQ_ALONG_LANE layout: (4608, 160, 4, 256)
        # rather than (4608, 64, 4, 640).
        #
        # Exists for `v3_native`, which skips the per-call layout conversion.
        # That conversion is ~86% of v3's measured decode time on the standard
        # spec, and a deployed system holding the cache in v3 layout would never
        # pay it -- so this spec measures steady state, while `decode_f8_kv9216`
        # measures drop-in migration cost. Both are worth reporting.
        name='decode_f8_kv9216_v3layout',
        seq_lens=[(1, 9216)] * 128,
        num_heads=128,
        lkv_dim=512,
        r_dim=64,
        page_size=256,
        q_dtype=jnp.float8_e4m3fn,
        kv_dtype=jnp.float8_e4m3fn,
        num_pages=128,
        tags=('primary', 'forward_only'),
        cache_layout='v3',
    ),
)

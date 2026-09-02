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

"""Utility functions for Batched Multi-Head Latent Attention (MLA)."""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def get_tpu_num_lanes() -> int:
  """Returns physical TPU vector lanes (defaults to 128 on CPU/test runner)."""
  try:
    return pltpu.get_tpu_info().num_lanes
  except (ValueError, AttributeError):
    return 128


def get_tpu_smem_capacity_bytes() -> int:
  """Returns SMEM capacity in bytes (defaults to 16MB on CPU/test runner)."""
  try:
    return pltpu.get_tpu_info().smem_capacity_bytes
  except (ValueError, AttributeError):
    return 16 * 1024 * 1024


def align_to(a, b):
  """Returns 'a' aligned up to the nearest multiple of 'b'."""
  return pl.cdiv(a, b) * b


def broadcast_minor(src, shape):
  """Broadcasts 'src' to 'shape' in the minor dimension."""
  if src.shape == shape:
    return src
  assert src.shape[:-1] == shape[:-1]
  if src.shape[-1] == 1:
    return jnp.broadcast_to(src, shape)
  num_lanes = get_tpu_num_lanes()
  assert src.shape[-1] % num_lanes == 0
  target_minor = align_to(shape[-1], src.shape[-1])
  reps = (1,) * (src.ndim - 1) + (target_minor // src.shape[-1],)
  broadcasted = jnp.tile(src, reps)
  return broadcasted[..., : shape[-1]]


def get_dtype_packing(dtype):
  """Returns number of packed elements per 32-bit word."""
  return 32 // jax.dtypes.itemsize_bits(dtype)


def transpose_kv_cache_to_v3(
    cache_kv: jax.Array,
    kv_packing: int | None = None,
) -> jax.Array:
  """Transposes untransposed 4D KV cache [pages, page_size // P, P, kv_dim]

  to transposed 4D KV cache [pages, kv_dim // P, P, page_size].
  """
  total_num_pages, page_size_per_packing, packing, kv_dim = cache_kv.shape
  if kv_packing is not None:
    assert kv_packing == packing
  page_size = page_size_per_packing * packing
  flat_tokens = cache_kv.reshape((total_num_pages, page_size, kv_dim))
  transposed_2d = flat_tokens.transpose((0, 2, 1))
  return transposed_2d.reshape(
      (total_num_pages, kv_dim // packing, packing, page_size)
  )


def transpose_kv_cache_from_v3(
    transposed_cache_kv: jax.Array,
    kv_packing: int | None = None,
) -> jax.Array:
  """Transposes 4D transposed KV cache [pages, kv_dim // P, P, page_size]

  back to untransposed 4D KV cache [pages, page_size // P, P, kv_dim].
  """
  total_num_pages, kv_sublanes, packing, page_size = transposed_cache_kv.shape
  if kv_packing is not None:
    assert kv_packing == packing
  kv_dim = kv_sublanes * packing
  flat_channels = transposed_cache_kv.reshape(
      (total_num_pages, kv_dim, page_size)
  )
  untransposed_2d = flat_channels.transpose((0, 2, 1))
  return untransposed_2d.reshape(
      (total_num_pages, page_size // packing, packing, kv_dim)
  )

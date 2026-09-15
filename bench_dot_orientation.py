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

"""Isolates whether MLA v3's KV layout costs it on the MXU.

v3 stores the paged cache SEQ_ALONG_LANE -- tokens on lanes, kv_dim on
sublanes -- where v2 puts tokens on sublanes. That flips the orientation of
both dots in flash attention, and in opposite directions:

                QK                              PV
  v2   [M,640] x [S,640]  contract (1,1)  NT    [M,S] x [S,512]  contract (1,0)  NN
  v3   [M,640] x [640,S]  contract (1,0)  NN    [M,S] x [512,S]  contract (1,1)  NT

So each pays the transposed form on exactly one dot. On the dense
`chunked_prefill_f8_kv8192` shape v3's kernel is 30% slower than v2's, and the
PV orientation is a candidate -- but QK is the larger dot (640 of 1152
contraction elements), so a uniform NT penalty would hurt v2 *more*. Reasoning
cannot settle that; this measures it.

Shapes match one `q_split=16` chunk of that spec: M=256 rows, S=2048 KV,
contraction 640 for QK and 2048 for PV, output 512 for PV.

Timing is a `fori_loop` of `reps` dots accumulating into one output, so the
dots cannot be CSE'd away and the loop is dominated by the matmul (the f32 add
on [256,512] is ~0.5 MB against a ~0.7 GFLOP dot).
"""

import functools
import time

import jax
import jax.numpy as jnp
from jax import lax

_FP8 = jnp.float8_e4m3fn
_REPS = 200


def _time(fn, *args, iters=10):
  fn(*args)[0].block_until_ready()
  ts = []
  for _ in range(iters):
    t0 = time.perf_counter()
    fn(*args)[0].block_until_ready()
    ts.append(time.perf_counter() - t0)
  return sorted(ts)[len(ts) // 2] * 1e3


def _make(lhs_shape, rhs_shape, contract, out_shape):
  @jax.jit
  def run(a, b):
    def body(_, acc):
      d = lax.dot_general(
          a, b, (contract, ((), ())), preferred_element_type=jnp.float32
      )
      return acc + d

    out = lax.fori_loop(0, _REPS, body, jnp.zeros(out_shape, jnp.float32))
    return (out,)

  return run


def main():
  key = jax.random.PRNGKey(0)
  m, s, kv_dim, d = 256, 2048, 640, 512

  def rand(shape, dtype):
    nonlocal key
    key, sub = jax.random.split(key)
    return jax.random.uniform(sub, shape, dtype=jnp.bfloat16).astype(dtype)

  cases = []

  # ---- QK: contraction 640, output [M, S] ---------------------------------
  q = rand((m, kv_dim), _FP8)
  cases.append((
      "QK  NN  (v3: kv_dim on sublanes)",
      _make((m, kv_dim), (kv_dim, s), (((1,), (0,))), (m, s)),
      q, rand((kv_dim, s), _FP8),
      2 * m * s * kv_dim,
  ))
  cases.append((
      "QK  NT  (v2: tokens on sublanes)",
      _make((m, kv_dim), (s, kv_dim), (((1,), (1,))), (m, s)),
      q, rand((s, kv_dim), _FP8),
      2 * m * s * kv_dim,
  ))

  # ---- PV: contraction S, output [M, d] -----------------------------------
  # `p` is f32 out of the softmax in both implementations.
  p = rand((m, s), jnp.float32)
  cases.append((
      "PV  NN  (v2: tokens on sublanes)",
      _make((m, s), (s, d), (((1,), (0,))), (m, d)),
      p, rand((s, d), _FP8),
      2 * m * s * d,
  ))
  cases.append((
      "PV  NT  (v3: tokens on lanes)",
      _make((m, s), (d, s), (((1,), (1,))), (m, d)),
      p, rand((d, s), _FP8),
      2 * m * s * d,
  ))

  print(f"M={m} S={s} kv_dim={kv_dim} d={d}, {_REPS} dots per timed call\n")
  for name, fn, a, b, flops in cases:
    try:
      ms = _time(fn, a, b)
      tflops = (flops * _REPS) / (ms * 1e-3) / 1e12
      print(f"  {name:36s} {ms:8.3f} ms   {tflops:7.1f} TFLOP/s")
    except Exception as exc:  # pylint: disable=broad-except
      print(f"  {name:36s} RAISED {type(exc).__name__}: {str(exc)[:70]}")


if __name__ == "__main__":
  main()

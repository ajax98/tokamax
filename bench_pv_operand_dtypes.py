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

"""Measures what the MXU actually does with mixed-precision PV operands.

`p_same_dtype_as_v` casts the softmax probabilities to the KV dtype before the
PV dot. It measures -16.1% on `chunked_prefill_f8_kv8192` and +9.4% on
`decode_f8_kv9216_p1024`, and the reason is not visible in the source: HLO
keeps `dot(f32, fp8)` as a single `dot_general` with no `convert`, so whatever
reconciles the two operand formats happens inside the TPU backend.

Two models fit those numbers and this tells them apart:

  A. The backend upconverts V (fp8 -> bf16) and runs bf16 x bf16. V is the
     *larger* operand -- [512, S] against P's [rows, S] -- so this path pays
     to convert 4x more data than casting P down would, and fp8 x fp8 should
     look markedly faster here.
  B. The MXU consumes bf16 x fp8 (or f32 x fp8) directly. Then nothing large
     is converted either way, and fp8 x fp8 should be at most modestly faster
     -- whatever the native fp8 rate advantage is, and no more.

Under A the `f32 x fp8` row is slow because of a hidden V conversion; under B
it is slow only in proportion to P's width. The `bf16 x fp8` row is the
discriminator: if mixed operands are genuinely supported it should sit close
to `fp8 x fp8`, and if they are not it should match `bf16 x bf16`.

Shapes are the two real ones, taken from `calculate_block_sizes`:

    decode_f8_kv9216_p1024    rows=128  S=3072   (bq_c_sz=1, kv=3, page=1024)
    chunked_prefill_kv8192    rows=256  S=2048   (bq_c_sz=2, kv=8, page=256)

Timing is a `fori_loop` of `reps` dots accumulating into one output, so the
dots cannot be CSE'd away and the loop is dominated by the matmul.
"""

import time

import jax
from jax import lax
import jax.numpy as jnp

_FP8 = jnp.float8_e4m3fn
_REPS = 200

# (label, P dtype, V dtype). `p_same_dtype_as_v=False` is the f32 row;
# `True` is the fp8 row. bf16 rows isolate where the cost sits.
_COMBOS = [
    ("f32  x fp8   (flag off)", jnp.float32, _FP8),
    ("fp8  x fp8   (flag on) ", _FP8, _FP8),
    ("bf16 x fp8   (mixed)   ", jnp.bfloat16, _FP8),
    ("bf16 x bf16  (baseline)", jnp.bfloat16, jnp.bfloat16),
]

_SHAPES = [
    ("decode_f8_kv9216_p1024", 128, 3072),
    ("chunked_prefill_kv8192", 256, 2048),
]


def _time(fn, *args, iters=10):
  fn(*args)[0].block_until_ready()
  ts = []
  for _ in range(iters):
    t0 = time.perf_counter()
    fn(*args)[0].block_until_ready()
    ts.append(time.perf_counter() - t0)
  return sorted(ts)[len(ts) // 2] * 1e3


def _make(out_shape):
  @jax.jit
  def run(p, v):
    def body(_, acc):
      # Matches flash_attention_pv: contract the S axis, accumulate in f32.
      d = lax.dot_general(
          p, v, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
      )
      return acc + d

    return (lax.fori_loop(0, _REPS, body, jnp.zeros(out_shape, jnp.float32)),)

  return run


def main():
  key = jax.random.PRNGKey(0)
  d_nope = 512

  def rand(shape, dtype):
    nonlocal key
    key, sub = jax.random.split(key)
    return jax.random.uniform(sub, shape, dtype=jnp.bfloat16).astype(dtype)

  for name, rows, s in _SHAPES:
    flops = 2 * rows * s * d_nope
    print(f"\n{name}:  P=[{rows},{s}]  V=[{d_nope},{s}]  "
          f"{_REPS} dots per timed call")
    base = None
    for label, p_dt, v_dt in _COMBOS:
      fn = _make((rows, d_nope))
      try:
        ms = _time(fn, rand((rows, s), p_dt), rand((d_nope, s), v_dt))
      except Exception as exc:  # pylint: disable=broad-except
        print(f"    {label}  RAISED {type(exc).__name__}: {str(exc)[:60]}")
        continue
      tflops = (flops * _REPS) / (ms * 1e-3) / 1e12
      if base is None:
        base = ms
      print(f"    {label}  {ms:8.3f} ms  {tflops:7.1f} TFLOP/s"
            f"  {base / ms:5.2f}x vs flag-off")

  print("\nReading it: if `bf16 x fp8` tracks `fp8 x fp8`, the MXU takes mixed")
  print("operands and model B holds. If it tracks `bf16 x bf16`, the fp8 side")
  print("is being widened and model A holds -- and the flag's real job is")
  print("avoiding a conversion of V, the larger operand.")


if __name__ == "__main__":
  main()

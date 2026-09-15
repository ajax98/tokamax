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

"""Verifies `q_split` against the reference.

`q_split` sets `bq_c_sz = bq_sz // q_split`, so `chunked_flash_attention` walks
the query block in sub-chunks and each one carries its own `bq_start` into the
causal mask. An off-by-one there shifts the diagonal for every chunk but the
first. The kernel additionally defers each chunk's PV by one
iteration, so the two interact and are checked together.

Deliberately run on *small* shapes. `check_prefill_bf16.py` uses
`[(2048, 2048)] * 2`, where the pure-JAX reference costs ~20 minutes per
invocation and dominates everything -- 3 of 14 cases completed in 70 minutes.
None of what `q_split` can get wrong depends on absolute sequence length: it is
determined by `bq_sz`, the split count, and the mask offset arithmetic. A
512-token prompt exercises identical code roughly 16x faster, so this can
actually be run as a gate rather than admired from a distance.

Both prefill regimes are covered, because they mask differently:
  * whole-prompt (`q_len == kv_len`)   -- the diagonal runs through the block
  * chunked      (`kv_len > q_len`)    -- every query token also sees history
"""

import numpy as np

import jax
import jax.numpy as jnp

from tokamax._src.ops.experimental.mla import api as mla_api
from tokamax._src.ops.experimental.mla import test_base
from tokamax._src.ops.experimental.mla import v3_op
from tokamax._src.ops.experimental.mla.v2 import kernel as kernel_v2

_BF16 = jnp.bfloat16
_FP8 = jnp.float8_e4m3fn


def _check(name, seq_lens, dtype, atol, rtol, cases):
  (
      ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
      kv_lens, page_indices, cu_q_lens, distribution,
  ) = test_base.generate_mla_inputs(
      seq_lens, 128, 512, 64, 256, dtype, dtype, 128,
      rng=np.random.default_rng(1234),
  )
  expected, _ = kernel_v2.ref_mla_ragged_paged_attention(
      ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv.copy(),
      kv_lens, page_indices, cu_q_lens, distribution,
  )
  e = np.asarray(expected, dtype=np.float32)
  print(f"\n=== {name}: seq_lens={seq_lens} out={e.shape} ===")

  for kv, q, knobs in cases:
    op = mla_api.IMPLEMENTATIONS["v3"].replace(
        config=v3_op.Config(
            num_kv_pages_per_block=kv,
            num_queries_per_block=q,
            batch_size=1,
            n_buffer=2,
            vmem_limit_bytes=64 * 1024 * 1024,
            **knobs,
        )
    )

    @jax.jit
    def run(c, op=op):
      return op(
          ql_nope=ql_nope, q_pe=q_pe, new_kv_c=new_kv_c, new_k_pe=new_k_pe,
          cache_kv=c, kv_lens=kv_lens, page_indices=page_indices,
          cu_q_lens=cu_q_lens, distribution=distribution,
      )

    desc = ",".join(f"{k}={v}" for k, v in sorted(knobs.items())) or "-"
    try:
      a = np.asarray(run(cache_kv.copy())[0], dtype=np.float32)
      bad = ~np.isclose(a, e, atol=atol, rtol=rtol)
      frac = bad.sum() / bad.size
      tok = np.flatnonzero(bad.reshape(bad.shape[0], -1).sum(1))
      # These are pure reassociation, not precision trades, so the bar is
      # essentially exact rather than a mismatch budget.
      verdict = "OK " if frac <= 1e-4 else "BAD"
      extra = "" if tok.size == 0 else f" first_bad_token={tok[0]}"
      print(
          f"  kv={kv:<2d} q={q:<3d} {verdict} {int(bad.sum()):8d}/{bad.size}"
          f" ({frac:.4%}) maxdiff={np.abs(a - e).max():.4f}{extra}  {desc}"
      )
    except Exception as exc:  # pylint: disable=broad-except
      print(f"  kv={kv:<2d} q={q:<3d} RAISED {type(exc).__name__}:"
            f" {str(exc)[:70]}  {desc}")


def main():
  cases = [
      (2, 32, {}),
      (2, 32, dict(q_split=2)),
      (2, 32, dict(q_split=4)),
      (2, 32, dict(q_split=8)),
      (2, 32, dict(q_split=32)),
      (2, 32, dict(q_split=8)),
      (2, 32, dict(q_split=4)),
      (2, 16, dict(q_split=4)),
      # `kv_slack_pad_lanes=0` is the tuned prefill setting. It is pure
      # padding, no arithmetic change -- but it shrinks the KV staging buffer
      # to `bkv_sz + 2*page_size`, and the stitch writes into that slack, so a
      # sizing error would corrupt KV rather than merely slow things down.
      # Note it is the *opposite* of decode's tuned value (128), where 0 gave a
      # power-of-two stride that aliased VMEM banks.
      (2, 32, dict(q_split=8, kv_slack_pad_lanes=0)),
      (2, 32, dict(kv_slack_pad_lanes=0)),
      # The numeric knobs, isolated. Previously only ever measured as a bundle
      # -- `narrow_scores + narrow_softmax + p_same_dtype_as_v +
      # kv_slack_pad_lanes=0` -- which was 4% faster and 17.99% wrong, and the
      # bundle was dropped without establishing which member was responsible.
      # `p_same_dtype_as_v` casts the softmax probabilities to the KV dtype,
      # which is the obvious suspect; `narrow_softmax` only narrows `s - m` and
      # the `exp`, where the values are bounded. Worth separating, because the
      # score tile is ~5 VALU passes over 1 MB per chunk and is plausibly half
      # the kernel.
      (2, 32, dict(narrow_scores=True, narrow_softmax=True, q_split=8)),
      (2, 32, dict(narrow_scores=True, q_split=8)),
      (2, 32, dict(p_same_dtype_as_v=True, q_split=8)),
      # The tuned chunked-prefill-fp8 settings. These are v2's *defaults*
      # (s_dtype=bf16, p_same_dtype_as_v=True) and are worth 17.9% on
      # chunked_prefill_f8_kv8192, but the same trio measured 17.99% WRONG on
      # prefill_bf16 -- bf16 data checked at 2e-2 has no room for a bf16
      # `s - m`. fp8 data is checked at 1e-1/2e-1, the budget v2 itself needs,
      # so they should be admissible here. "Should" is why this is measured.
      (8, 32, dict(p_same_dtype_as_v=True, q_split=16)),
      (8, 32, dict(narrow_scores=True, narrow_softmax=True,
                   p_same_dtype_as_v=True, q_split=16)),
  ]
  # Whole-prompt prefill, the regime where q_split was measured.
  _check("prefill bf16 (q_len == kv_len)", [(512, 512)] * 2, _BF16,
         2e-2, 2e-2, cases)
  # Chunked prefill: 768 tokens of history, so the causal predicate is offset
  # rather than running through the block. fp8, hence the looser bound.
  _check("chunked prefill fp8 (kv_len > q_len)", [(256, 1024)], _FP8,
         1e-1, 2e-1, cases)
  # The dense shape the precision knobs were tuned on: 7936 cached tokens, so
  # every query row sums over nearly all 8192 KV. That is the worst case for a
  # narrowed accumulator chain, which is exactly why it is worth checking here
  # rather than inferring from the 1024 result.
  _check("chunked prefill fp8 dense (kv=8192)", [(256, 8192)], _FP8,
         1e-1, 2e-1, [c for c in cases if c[0] == 8])


if __name__ == "__main__":
  main()

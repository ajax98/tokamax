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

"""Checks v3 against the reference on the `prefill_bf16` shape.

`mla_v2_v3_bench_test.py` only times this workload; nothing verifies its
numerics. That was tolerable while prefill was not being tuned, but the stitch
path on prefill has just been fixed and the config is about to be swept, so a
fast config that is silently wrong is a live risk -- exactly the failure mode
that the `num_queries_per_block=1` bug represented.

Whole-prompt prefill: q_len == kv_len == 2048 over 2 sequences, so there is no
cached history and every KV token is a new one. That is the opposite extreme
from decode (one new token) and therefore the case most likely to expose the
stitch, which is why it is worth checking at several block sizes rather than
just the shipped one.
"""

import numpy as np

import jax
import jax.numpy as jnp

from tokamax._src.ops.experimental.mla import api as mla_api
from tokamax._src.ops.experimental.mla import test_base
from tokamax._src.ops.experimental.mla import v3_op
from tokamax._src.ops.experimental.mla.v2 import kernel as kernel_v2

_BF16 = jnp.bfloat16
# bf16 has 8 mantissa bits; the reference accumulates in f32. These are the
# same bounds the op tests use for the bf16 shapes.
_ATOL, _RTOL = 2e-2, 2e-2
_MAX_BAD_FRAC = 1e-4


def main():
  seq_lens = [(2048, 2048)] * 2
  (
      ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
      kv_lens, page_indices, cu_q_lens, distribution,
  ) = test_base.generate_mla_inputs(
      seq_lens, 128, 512, 64, 256, _BF16, _BF16, 128,
      rng=np.random.default_rng(1234),
  )
  print(f"ql_nope={ql_nope.shape} cache_kv={cache_kv.shape}")
  print(f"kv_lens={np.asarray(kv_lens)} cu_q_lens={np.asarray(cu_q_lens)}")
  print(f"distribution={np.asarray(distribution)}")

  expected, _ = kernel_v2.ref_mla_ragged_paged_attention(
      ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv.copy(),
      kv_lens, page_indices, cu_q_lens, distribution,
  )
  e = np.asarray(expected, dtype=np.float32)

  # `impl` matters: v3_native needs a v3-layout cache, which `generate_mla_inputs`
  # does not produce, so this exercises `v3` (same kernel, plus a transpose).
  # The last entry is the tuned prefill configuration. Unlike the block-size
  # rows above it changes *numerics*: `narrow_softmax` runs the subtract and
  # the exp at bf16, and `p_same_dtype_as_v` feeds the PV dot a bf16 `p`. Both
  # are measured wins here and both need checking rather than assuming, since
  # the block-size rows say nothing about them.
  cases = [
      (8, 16, {}), (4, 16, {}), (2, 16, {}), (4, 8, {}), (4, 32, {}),
      (8, 1, {}),
      (2, 32, dict(narrow_scores=True, narrow_softmax=True,
                   p_same_dtype_as_v=True, kv_slack_pad_lanes=0)),
      (2, 32, dict(narrow_scores=True, narrow_softmax=True)),
      (2, 32, {}),
      # `q_split` sub-chunks the query block, so each chunk carries its own
      # `bq_start` into the causal mask. An off-by-one there shifts the
      # diagonal for every chunk but the first, which is why these are checked
      # at several splits rather than one -- and with `two_step`, which
      # additionally defers each chunk's PV by one.
      (2, 32, dict(q_split=2)),
      (2, 32, dict(q_split=4)),
      (2, 32, dict(q_split=8)),
      (2, 32, dict(q_split=32)),
      (2, 32, dict(q_split=8)),
  ]
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

    try:
      actual, _ = run(cache_kv.copy())
      a = np.asarray(actual, dtype=np.float32)
      bad = ~np.isclose(a, e, atol=_ATOL, rtol=_RTOL)
      frac = bad.sum() / bad.size
      tok = np.flatnonzero(bad.reshape(bad.shape[0], -1).sum(1))
      verdict = "OK " if frac <= _MAX_BAD_FRAC else "BAD"
      extra = "" if tok.size == 0 else f"  first_bad_token={tok[0]} n_tok={tok.size}"
      print(
          f"  kv={kv:<3d} q={q:<3d} {verdict} {int(bad.sum()):9d}/{bad.size}"
          f" ({frac:.4%}) maxdiff={np.abs(a - e).max():.4f}{extra}"
          f"  {','.join(sorted(knobs)) if knobs else '-'}"
      )
    except Exception as exc:  # pylint: disable=broad-except
      print(f"  kv={kv:<3d} q={q:<3d} RAISED {type(exc).__name__}: {str(exc)[:70]}")


if __name__ == "__main__":
  main()

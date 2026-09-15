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

"""Writes the hand-swept MLA configs into tokamax's autotuning cache.

Every config in this file was found by hand sweep with
`run_mla_profile.py --variant` and ranked on device occupancy. They are
recorded in `mla_v2_v3_bench_test.py`, but that is documentation: nothing reads
it. The place a config has to live to reach a caller is
`tokamax/data/autotuning/<device_kind>/<op>.json`, which `Op.__call__` consults
before falling back to `_get_heuristics_config`.

Without this, a caller gets the heuristic block sizes plus whatever the
`Config` field defaults happen to be -- so none of `q_split`, `q=32`, `kv=32`,
`kv=36`, or the per-shape `kv_slack_pad_lanes` polarity. The
autotuner cannot find them either: `_get_autotuning_configs` sweeps only four
axes and its ranges stop at `num_queries_per_block=16`.

Rather than hand-build cache entries (the keys are `BoundArguments` with
abstractified arguments, and getting that wrong yields a file that silently
never matches), this runs the real `tokamax.autotune` with the candidate set
pinned to a single config per spec. That is fast -- one measurement each -- and
produces keys identical to what a normal call looks up.

Usage:
  python record_best_configs.py --out-dir /mnt/nfs/ajaygopi/autotune_results
  # then: cp <out-dir>/autotuning_cache/*.json tokamax/data/autotuning/tpu7x/
"""

import argparse
import contextlib
import dataclasses
import pathlib
import sys
import unittest.mock as mock

import tokamax
from tokamax._src.ops.experimental.mla import api as mla_api
from tokamax._src.ops.experimental.mla import arg_specs as mla_specs

from run_mla_autotune import abstractify_result
from run_mla_autotune import cache_filename
from run_mla_autotune import materialize

_MB = 1024 * 1024

# (spec name, impl, config overrides). Anything not named takes the `Config`
# field default, which for v3 already carries the mechanism-general flags
# unconditionally, leaving only kv_slack_pad_lanes=128 as a default here.
#
# v3 entries key off the `_v3layout` specs because `v3_native` consumes a
# SEQ_ALONG_LANE cache; that is the shape a caller of this op actually passes.
BEST = [
    # ---- decode: bq_sz is 1, so q_split/two_step are inert -----------------
    ("decode_bf16", "v2",
     dict(num_kv_pages_per_block=8, num_queries_per_block=1,
          decode_batch_size=1, mixed_q_split=1)),
    ("decode_bf16_v3layout", "v3_native",
     dict(num_kv_pages_per_block=8, num_queries_per_block=1,
          batch_size=4, n_buffer=2)),

    ("decode_f8", "v2",
     dict(num_kv_pages_per_block=8, num_queries_per_block=1,
          decode_batch_size=8, mixed_q_split=1)),
    # kv=32 makes bkv_sz == kv_len: one k-block per sequence, so the schedule
    # emits 128 tasks rather than 512. Worth 6% here and it flips the spec.
    ("decode_f8_v3layout", "v3_native",
     dict(num_kv_pages_per_block=32, num_queries_per_block=1,
          batch_size=2, n_buffer=2)),

    ("decode_f8_kv9216", "v2",
     dict(num_kv_pages_per_block=12, num_queries_per_block=1,
          decode_batch_size=8, mixed_q_split=1)),
    ("decode_f8_kv9216_v3layout", "v3_native",
     dict(num_kv_pages_per_block=36, num_queries_per_block=1,
          batch_size=2, n_buffer=2)),

    # At page_size 1024 there are 4x fewer pages, so the whole-sequence block
    # (kv=9) is a wash and the smaller one wins on kernel time instead.
    ("decode_f8_kv9216_p1024", "v2",
     dict(num_kv_pages_per_block=3, num_queries_per_block=1,
          decode_batch_size=8, mixed_q_split=1)),
    ("decode_f8_kv9216_p1024_v3layout", "v3_native",
     dict(num_kv_pages_per_block=3, num_queries_per_block=1,
          batch_size=4, n_buffer=2)),

    # ---- prefill: q_split and two_step are live ----------------------------
    # Whole-prompt prefill inverts two decode defaults -- every KV token is new,
    # so pad=0 beats pad=128. This spec also wanted gate_stitch=False (-3.1%),
    # which is no longer expressible now that the stitch gate is unconditional.
    ("prefill_bf16", "v2",
     dict(num_kv_pages_per_block=8, num_queries_per_block=16,
          decode_batch_size=1, mixed_q_split=4)),
    ("prefill_bf16_v3layout", "v3_native",
     dict(num_kv_pages_per_block=2, num_queries_per_block=32,
          batch_size=1, n_buffer=2, q_split=8, kv_slack_pad_lanes=0)),

    ("chunked_prefill_f8_kv1024", "v2",
     dict(num_kv_pages_per_block=4, num_queries_per_block=32,
          decode_batch_size=1, mixed_q_split=4)),
    ("chunked_prefill_f8_kv1024_v3layout", "v3_native",
     dict(num_kv_pages_per_block=4, num_queries_per_block=32,
          batch_size=1, n_buffer=2, q_split=8, p_same_dtype_as_v=True)),

    # History-dominated, so the *decode* polarity for pad: 128.
    # q_split=16 is what makes kv=8 fit; at q_split=8 it exceeds VMEM.
    ("chunked_prefill_f8_kv8192", "v2",
     dict(num_kv_pages_per_block=8, num_queries_per_block=32,
          decode_batch_size=1, mixed_q_split=4)),
    # p_same_dtype_as_v is worth 16.1% here and omitting it was a real bug:
    # this list disagreed with mla_v2_v3_bench_test.py, which is the file that
    # recorded the timings. A re-trace without it came back 0.841x against v2
    # instead of 1.002x.
    ("chunked_prefill_f8_kv8192_v3layout", "v3_native",
     dict(num_kv_pages_per_block=8, num_queries_per_block=32,
          batch_size=1, n_buffer=2, q_split=16,
          p_same_dtype_as_v=True)),
]


@contextlib.contextmanager
def _pinned_configs(op, config):
  """Restricts `op`'s autotuning search to exactly `config`.

  Patched on the *type* rather than the instance because `Op` is a frozen
  pydantic dataclass and does not accept attribute assignment.
  """
  with mock.patch.object(
      type(op), "_get_autotuning_configs", lambda self, ba: {config}
  ):
    yield


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out-dir", default="/mnt/nfs/ajaygopi/autotune_results")
  ap.add_argument("--timeout", type=float, default=600.0)
  opts = ap.parse_args(argv)

  by_name = {s.name: s for s in mla_specs.ARG_SPECS}
  by_impl = {}
  failures = []

  for spec_name, impl_name, overrides in BEST:
    spec = by_name[spec_name]
    op = mla_api.IMPLEMENTATIONS[impl_name]
    config = op.config_cls(vmem_limit_bytes=64 * _MB, **overrides)
    ba = op.bind(**materialize(spec.args))

    print(f"\n=== {spec_name} / {impl_name}", flush=True)
    print(f"    {config}", flush=True)
    try:
      with _pinned_configs(op, config):
        result = tokamax.autotune(
            [ba], ignore_cache=True, progress_bar=False, timeout=opts.timeout
        )
    except Exception as exc:  # pylint: disable=broad-except
      print(f"    RAISED {type(exc).__name__}: {str(exc)[:120]}")
      failures.append((spec_name, impl_name))
      continue

    # An entry that is an Exception means the config did not run; recording it
    # would poison the cache with a known-bad choice.
    bad = [
        e for _, data in result.data for e in data.values()
        if isinstance(e, Exception)
    ]
    if bad:
      print(f"    config failed to run: {str(bad[0])[:120]}")
      failures.append((spec_name, impl_name))
      continue

    by_impl[impl_name] = (
        result if impl_name not in by_impl else by_impl[impl_name] | result
    )
    print("    ok", flush=True)

  if not by_impl:
    print("nothing recorded", file=sys.stderr)
    return 1

  cache_dir = pathlib.Path(opts.out_dir) / "autotuning_cache"
  cache_dir.mkdir(parents=True, exist_ok=True)
  for impl_name, result in by_impl.items():
    path = cache_dir / cache_filename(mla_api.IMPLEMENTATIONS[impl_name])
    path.write_text(abstractify_result(result).dump_cache_str())
    n = sum(len(d) for _, d in result.data)
    print(f"wrote {path}  ({n} entries)")

  if failures:
    print(f"\n{len(failures)} did not record: {failures}", file=sys.stderr)
  print(f"\ninstall with:  cp {cache_dir}/*.json "
        "tokamax/data/autotuning/tpu7x/")
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())

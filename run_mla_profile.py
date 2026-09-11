#!/usr/bin/env python3
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
"""Capture xprof device traces for the tuned MLA v2 and v3 configs.

CDK only distributes profiles -- the capture is ours to implement. Traces must
land as `*.xplane.pb` under `$CDK_OUTPUT_DIR`, after which
`cdk job xprof <job-id>` prints a `go/cdk-xprof?<path>` link per file.

Each (spec, impl) is traced into its own subdirectory so the links are
self-describing rather than a pile of anonymous `.xplane.pb`s.

Two things this deliberately makes visible in the trace, because they are the
claims worth checking:

  * **v3's KV-cache layout conversion.** `v3_op._fwd` transposes the cache in
    and out on every call, and that is inside the measured region. It is ~3% on
    the 21 MB prefill caches but the decode spec's cache is 755 MB, where the
    transpose can plausibly dominate. Those ops appear as separate XLA fusions
    in the trace, so the split is readable directly.
  * **v3's extra schedule kernel.** v3 runs `mla_metadata_schedule` before the
    attention kernel, once per pass. Reported separately by op name.

Usage:
  python run_mla_profile.py --all
  python run_mla_profile.py --spec chunked_prefill_f8_kv8192 --iterations 20
"""

import argparse
import csv
import dataclasses
import functools
import json
import os
import pathlib
import statistics
import sys
import time
import traceback

import jax
import numpy as np

import tokamax
from tokamax._src.ops.experimental.mla import api as mla_api
from tokamax._src.ops.experimental.mla import arg_specs as mla_specs

from run_mla_autotune import IMPLS
from run_mla_autotune import SPEC_NAMES
from run_mla_autotune import materialize


def config_from_csv(op, csv_path: pathlib.Path):
  """Reads the winning config out of a sweep CSV written by run_mla_autotune.

  Deliberately not `ba.default_config`: that route loads
  `tokamax/data/autotuning/<device_kind>/<op>.json`, and cache files written by
  `dump_cache_str` for these ops currently fail to deserialize (the reconstruct
  path ends up calling the jaxtyped `_fwd` with `ShapeDtypeStruct`s and
  typeguard rejects them). A cache miss is only a *warning*, so going through
  `default_config` would silently profile the heuristics config instead of the
  tuned one -- exactly the failure this function exists to avoid.

  The CSV is already sorted best-first, with failures last.
  """
  with open(csv_path, newline="") as fh:
    rows = list(csv.DictReader(fh))
  ok = [r for r in rows if r["status"] == "OK"]
  if not ok:
    raise ValueError(f"no successful configs in {csv_path}")
  row = ok[0]

  fields = {f.name: f for f in dataclasses.fields(op.config_cls)}
  kwargs = {}
  for name, field in fields.items():
    # A sweep CSV is a snapshot of the Config as it was when the sweep ran.
    # Fields added to Config afterwards (the ported v2 knobs) have no column,
    # so fall back to the dataclass default rather than failing -- otherwise
    # every new knob invalidates every existing CSV.
    if name not in row:
      continue
    raw = row[name]
    if field.type is bool or raw in ("True", "False"):
      kwargs[name] = raw == "True"
    else:
      kwargs[name] = int(raw)
  return op.config_cls(**kwargs), float(row["median_ms"])


def profile_one(spec, impl_name, out_dir, *, iterations, warmup, configs_dir,
                overrides=None):
  """Traces one (spec, impl) pair; returns a summary dict."""
  op = mla_api.IMPLEMENTATIONS[impl_name]
  args = materialize(spec.args)
  ba = op.bind(**args)

  heuristics = ba.heuristics_config
  csv_path = pathlib.Path(configs_dir) / f"{spec.name}__{impl_name}.csv"
  if csv_path.exists():
    config, swept_median_ms = config_from_csv(op, csv_path)
  else:
    # New spec with no sweep yet. Start from heuristics; `--variant` supplies
    # the real configuration. Explicit rather than silent, because a heuristics
    # config profiled by accident looks like a legitimate result.
    print(f"  NOTE: no sweep CSV at {csv_path}; basing on heuristics config")
    config, swept_median_ms = heuristics, float("nan")

  # `--variant k=v,...` re-profiles the CSV winner with fields overridden. The
  # autotuner ranked configs using the copy-polluted metric above, so its choice
  # is suspect: v3_native picked kv=1 (1.563 ms kernel) where the converting
  # v3_op picked kv=8 (0.972 ms kernel) on the same workload. This lets us
  # re-check a handful of candidates under the fixed harness without paying for
  # a full 72-config re-sweep.
  label_suffix = ""
  if overrides:
    fields = {f.name: f for f in dataclasses.fields(op.config_cls)}
    kwargs = dataclasses.asdict(config)
    for item in overrides.split(","):
      k, v = item.split("=", 1)
      if k not in fields:
        raise ValueError(f"unknown config field {k!r} for {impl_name}")
      kwargs[k] = (v == "True") if v in ("True", "False") else int(v)
    config = op.config_cls(**kwargs)
    label_suffix = "__" + overrides.replace(",", "_").replace("=", "")
  used_tuned = config != heuristics

  print(f"\n{'=' * 70}")
  print(f"=== {spec.name} / {impl_name}")
  print(f"{'=' * 70}")
  print(f"  heuristics config : {heuristics}")
  print(f"  tuned config      : {config}")
  print(f"  swept median      : {swept_median_ms:.4f} ms")
  print(f"  differs from heur : {used_tuned}")
  sys.stdout.flush()

  tuned_op = op.replace(config=config)

  # `cache_kv` is donated, and the updated cache is chained into the next call.
  #
  # Without this, every iteration re-passes a buffer the caller still owns while
  # the kernel's `input_output_aliases` demands to write it in place, so XLA
  # inserts a full copy of the cache -- 1.13 ms on the 755 MB decode cache,
  # comparable to the attention kernel itself. Worse, whether that copy overlaps
  # the kernel or serializes is a scheduling decision that differs between
  # harnesses: the same v3_native config measured 1.554 ms under the autotuner
  # (copy hidden behind the kernel) and 3.116 ms here (copy serialized). The
  # metric is a disjoint interval union, so overlap silently halves it.
  #
  # Chaining output to input is also what a serving loop does -- each decode
  # step consumes the cache and produces the next one -- so this is the more
  # faithful measurement, not just the more stable one.
  other_args = {k: v for k, v in args.items() if k != "cache_kv"}

  @functools.partial(jax.jit, donate_argnames=("cache_kv",))
  def run(cache_kv, other):
    return tuned_op(cache_kv=cache_kv, **other)

  # Warm up outside the trace so compilation does not land in it.
  cache = args["cache_kv"]
  for _ in range(warmup):
    out, cache = run(cache, other_args)
  jax.block_until_ready((out, cache))

  # Untraced wall-clock, as a sanity check against the device totals.
  times_ms = []
  for _ in range(iterations):
    t0 = time.perf_counter()
    out, cache = run(cache, other_args)
    jax.block_until_ready((out, cache))
    times_ms.append((time.perf_counter() - t0) * 1e3)
  median_ms = statistics.median(times_ms)

  trace_dir = out_dir / "xprof" / f"{spec.name}__{impl_name}{label_suffix}"
  trace_dir.mkdir(parents=True, exist_ok=True)

  jax.profiler.start_trace(str(trace_dir))
  try:
    for _ in range(iterations):
      out, cache = run(cache, other_args)
    jax.block_until_ready((out, cache))
  finally:
    jax.profiler.stop_trace()

  xplanes = sorted(trace_dir.rglob("*.xplane.pb"))
  print(f"  wallclock median  : {median_ms:.4f} ms ({iterations} iters, untraced)")
  print(f"  xplane files      : {len(xplanes)}")
  for p in xplanes:
    print(f"    {p}")
  if not xplanes:
    print("  !!! no .xplane.pb written -- cdk job xprof will find nothing", file=sys.stderr)
  sys.stdout.flush()

  return dict(
      spec=spec.name,
      impl=impl_name + label_suffix,
      config=str(config),
      config_source="tuned-from-csv",
      swept_median_ms=swept_median_ms,
      wallclock_median_ms=round(median_ms, 4),
      wallclock_min_ms=round(min(times_ms), 4),
      n_xplane=len(xplanes),
      trace_dir=str(trace_dir),
  )


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--spec", action="append", choices=SPEC_NAMES)
  parser.add_argument("--all", action="store_true")
  parser.add_argument("--impl", action="append", choices=IMPLS)
  parser.add_argument("--iterations", type=int, default=10)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--out-dir", default=None)
  parser.add_argument(
      "--variant",
      action="append",
      help="Config overrides applied to the CSV winner, e.g."
           " 'num_kv_pages_per_block=8'. Repeatable; each is profiled"
           " separately.",
  )
  parser.add_argument(
      "--configs-from",
      default="/mnt/nfs/ajaygopi/autotune_results",
      help="Directory of <spec>__<impl>.csv sweep results.",
  )
  opts = parser.parse_args()

  if opts.all:
    spec_names = list(SPEC_NAMES)
  elif opts.spec:
    spec_names = opts.spec
  else:
    parser.error("pass --spec NAME (repeatable) or --all")
  impls = opts.impl or list(IMPLS)

  # Must be $CDK_OUTPUT_DIR: `cdk job xprof` scans
  # `gs://cloud-devkit/jobs/<id>/outputs` for `*.xplane.pb`, so a trace written
  # anywhere else is invisible to it.
  out_dir = pathlib.Path(
      opts.out_dir or os.environ.get("CDK_OUTPUT_DIR") or "_cdk_out"
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  devices = jax.devices()
  print(f"jax {jax.__version__}, {len(devices)} devices: {devices[:8]}")
  if not devices or devices[0].platform != "tpu":
    print("FATAL: profiling requires a TPU", file=sys.stderr)
    return 1
  print(f"device_kind: {devices[0].device_kind}")
  print(f"output dir:  {out_dir.resolve()}")

  by_name = {s.name: s for s in mla_specs.ARG_SPECS}
  summaries = []
  exit_code = 0

  for spec_name in spec_names:
    for impl_name in impls:
      for overrides in (opts.variant or [None]):
        try:
          summaries.append(
              profile_one(
                  by_name[spec_name],
                  impl_name,
                  out_dir,
                  iterations=opts.iterations,
                  warmup=opts.warmup,
                  configs_dir=opts.configs_from,
                  overrides=overrides,
              )
          )
        except Exception:  # pylint: disable=broad-except
          print(f"\n!!! {spec_name}/{impl_name} raised:", file=sys.stderr)
          traceback.print_exc()
          summaries.append(dict(spec=spec_name, impl=impl_name, error=True))
          exit_code = 1

  with open(out_dir / "profile_summary.json", "w") as fh:
    json.dump(summaries, fh, indent=2, default=str)

  print(f"\n{'=' * 70}")
  print("=== SUMMARY (untraced wallclock)")
  print(f"{'=' * 70}")
  for s in summaries:
    if s.get("error"):
      print(f"{s['spec']:28s} {s['impl']:3s}  RAISED")
      continue
    print(
        f"{s['spec']:28s} {s['impl']:3s}  wall={s['wallclock_median_ms']:9.4f} ms"
        f"  swept={s['swept_median_ms']:9.4f} ms  xplane={s['n_xplane']}"
    )
  print("\nRetrieve traces with:  cdk job xprof <job-id> -a")
  return exit_code


if __name__ == "__main__":
  sys.exit(main())

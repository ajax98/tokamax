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
"""Phase 1c: autotune the MLA v2 and v3 ops over the FP8 comparison arg-specs.

Emits, per (spec, impl):
  * `<spec>__<impl>.csv`  -- every candidate config with its timings, including
    the ones that failed (OOM shows up here, not as a crash).
  * `autotuning_cache/<op_name>.json` -- in the exact format tokamax reads back
    from `tokamax/data/autotuning/<device_kind>/`, so Phase 2 can run against a
    warm cache instead of re-tuning.
  * `<spec>__<impl>.result.json` -- the full `AutotuningResult`, reloadable with
    `tokamax.AutotuningResult.loads`.

Why `op.bind()` rather than `tokamax.get_bound_args()`: `get_bound_args` exists
to *discover* ops inside an arbitrary program, which it does by lowering to HLO
and scraping the result. That round-trip leaves the arguments abstract, so
callers then have to inject concrete arrays back in before anything can be
timed. We already know both the op and its arguments, so binding directly skips
the lowering and the re-injection along with it.

Usage:
  python run_mla_autotune.py --spec chunked_prefill_f8_kv8192
  python run_mla_autotune.py --all
"""

import argparse
import csv
import dataclasses
import json
import os
import pathlib
import re
import statistics
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

import tokamax
from tokamax._src.ops import op as op_lib
from tokamax._src.ops.experimental.mla import api as mla_api
from tokamax._src.ops.experimental.mla import arg_specs as mla_specs


def cache_filename(op) -> str:
  """The filename tokamax looks for when loading this op's autotuning cache.

  Must match `AutotuningCache._load_cache`, which snake-cases the op class name
  and reads `tokamax/data/autotuning/<device_kind>/<that>.json`. Getting it
  wrong produces a silent cache miss and a fallback to heuristics, not an error,
  so it is derived here rather than hardcoded.
  """
  return re.sub(r"(?!^)([A-Z])", r"_\1", type(op).__name__).lower() + ".json"


def abstractify_result(result):
  """Replaces concrete argument arrays with their abstract shapes/dtypes.

  `AutotuningResult.dump_cache_str` indexes by `ba.arguments`, i.e. it uses the
  arguments as a dict key, so they have to be hashable. The usual producer is
  `get_bound_args`, which recovers arguments from a lowering and therefore hands
  back abstract values that hash fine. We bind concrete arrays instead -- which
  autotuning requires, since it has to actually execute each config -- so the
  dump raises `TypeError: unhashable type: 'jaxlib._jax.ArrayImpl'` unless we
  abstractify first.

  This mirrors what `Op.__call__` does before serializing an opspec into the
  HLO (`op.py:232`), so the resulting cache keys match what a normal call
  produces at lookup time. That match is the whole point: a key built from
  concrete arrays would never be found again.
  """
  data = tuple(
      (ba.replace(arguments=op_lib._abstractify(dict(ba.arguments))), entry)  # pylint: disable=protected-access
      for ba, entry in result.data
  )
  return dataclasses.replace(result, data=data)


# Specs this harness will sweep. Hand-maintained rather than derived from
# `arg_specs.ARG_SPECS`, because `--all` walks this list and the full set is far
# more than is useful to sweep -- but that means a spec added to `arg_specs`
# without being added here fails as an argparse "invalid choice", not as a
# missing spec, which reads like a typo.
#
# The FP8 group came first, for the v2-vs-v3 decode comparison; the 8192 one is
# the only shape there with substantial cached history. `prefill_bf16` is the
# bf16 whole-prompt prefill shape (q_len == kv_len == 2048, so no history at
# all) and its `_v3layout` twin exists so `v3_native` can run it without paying
# a per-call cache transpose that v2 never pays.
SPEC_NAMES = (
    "prefill_bf16",
    "prefill_bf16_v3layout",
    "decode_bf16",
    "decode_bf16_v3layout",
    "decode_f8",
    "decode_f8_v3layout",
    "chunked_prefill_f8_kv8192_v3layout",
    "prefill_bf16_v3compact",
    "chunked_prefill_f8_kv8192",
    "chunked_prefill_f8_kv1024_v3layout",
    "decode_f8_kv9216",
    "chunked_prefill_f8_kv1024",
    "decode_f8_kv9216_v3layout",
    "decode_f8_kv9216_v3compact",
    "decode_f8_kv9216_p1024",
    "decode_f8_kv9216_p1024_v3layout",
    "decode_f8_kv9216_p1024_v3compact",
)
IMPLS = ("v2", "v3", "v3_native")

_FLOAT_GEN_DTYPES = (
    jnp.dtype(jnp.float32),
    jnp.dtype(jnp.bfloat16),
    jnp.dtype(jnp.float16),
)


def materialize(spec_args, seed: int = 1234):
  """Turns an ArgSpec's `ShapeDtypeStruct`s into real arrays.

  `jax.random.uniform` has no FP8 path, so FP8 arrays are generated in bfloat16
  and cast -- the same thing `test_base.generate_mla_inputs` does, which keeps
  the value distribution consistent with the correctness tests.
  """
  key = jax.random.PRNGKey(seed)
  out = {}
  for name, value in spec_args.items():
    if isinstance(value, jax.ShapeDtypeStruct):
      key, subkey = jax.random.split(key)
      dtype = jnp.dtype(value.dtype)
      gen_dtype = dtype if dtype in _FLOAT_GEN_DTYPES else jnp.bfloat16
      out[name] = jax.random.uniform(
          subkey, value.shape, dtype=gen_dtype, minval=0.0, maxval=1.0
      ).astype(dtype)
    else:
      # kv_lens / page_indices / cu_q_lens / distribution arrive as concrete
      # `HashableNPArray`s and must keep their exact values -- they define the
      # ragged structure the schedule is built from.
      out[name] = jnp.asarray(value)
  return out


def tune_one(spec, impl_name, out_dir, *, timeout, max_workers):
  """Autotunes one (spec, impl) pair. Returns a summary dict."""
  op = mla_api.IMPLEMENTATIONS[impl_name]
  args = materialize(spec.args)

  # `bind` applies defaults, so every `_fwd` parameter is present. The config is
  # *not* among them -- it lives on `ba.op.config` and starts as None, which is
  # precisely what autotuning fills in.
  ba = op.bind(**args)
  n_configs = len(ba.autotuning_configs)

  print(f"\n{'=' * 70}")
  print(f"=== {spec.name} / {impl_name}: {n_configs} candidate configs")
  print(f"{'=' * 70}", flush=True)

  start = time.perf_counter()
  # `ignore_cache=True` so a previous run's entry doesn't short-circuit the
  # sweep -- we want every config measured, not just the cached winner.
  result = tokamax.autotune(
      [ba],
      ignore_cache=True,
      progress_bar=False,
      timeout=timeout,
      max_workers=max_workers,
  )
  wall_s = time.perf_counter() - start

  rows = []
  ok = 0
  failed = 0
  oom = 0
  best = None

  for bound, data in result.data:
    del bound
    for config, entry in data.items():
      cfg_fields = dataclasses.asdict(config)
      if isinstance(entry, Exception):
        text = str(entry)
        # VMEM exhaustion is an expected, informative outcome here rather than
        # a harness failure -- v3's footprint grows with `num_queries_per_block`
        # and a chunk of its search space does not fit.
        is_oom = "RESOURCE_EXHAUSTED" in text or "Ran out of memory" in text
        oom += is_oom
        failed += 1
        rows.append(
            cfg_fields
            | dict(
                status="OOM" if is_oom else "ERROR",
                median_ms="",
                times_ms="",
                error=text.splitlines()[0][:300],
            )
        )
        continue

      ok += 1
      times = list(entry.evaluation_times_ms)
      median = statistics.median(times) if times else float("nan")
      rows.append(
          cfg_fields
          | dict(
              status="OK",
              median_ms=median,
              times_ms=";".join(f"{t:.6f}" for t in times),
              error="",
          )
      )
      if best is None or median < best[0]:
        best = (median, cfg_fields)

  if rows:
    # Config fields differ between v2 and v3, so each pair gets its own file
    # rather than a union schema full of blanks.
    field_order = [k for k in rows[0] if k not in ("status", "median_ms", "times_ms", "error")]
    columns = field_order + ["status", "median_ms", "times_ms", "error"]
    csv_path = out_dir / f"{spec.name}__{impl_name}.csv"
    with open(csv_path, "w", newline="") as fh:
      writer = csv.DictWriter(fh, fieldnames=columns)
      writer.writeheader()
      for row in sorted(rows, key=lambda r: (r["status"] != "OK", r["median_ms"] if r["median_ms"] != "" else 9e9)):
        writer.writerow(row)
    print(f"  wrote {csv_path}")

  with open(out_dir / f"{spec.name}__{impl_name}.result.json", "w") as fh:
    fh.write(result.dumps(prune_errors=True))

  print(f"  configs: {ok} ok / {failed} failed (of which OOM: {oom})")
  print(f"  autotune wall time: {wall_s:.1f}s")
  if best is not None:
    print(f"  BEST {best[0]:.6f} ms  {best[1]}")
  else:
    print("  BEST: none -- every config failed")
  sys.stdout.flush()

  return dict(
      spec=spec.name,
      impl=impl_name,
      n_configs=n_configs,
      ok=ok,
      failed=failed,
      oom=oom,
      wall_s=round(wall_s, 1),
      best_median_ms=best[0] if best else None,
      best_config=best[1] if best else None,
      result=result,
  )


def rebuild_cache(src_dir: pathlib.Path, out_dir: pathlib.Path) -> int:
  """Regenerates autotuning cache files from saved `.result.json` files."""
  cache_dir = out_dir / "autotuning_cache"
  cache_dir.mkdir(parents=True, exist_ok=True)

  by_impl = {}
  for path in sorted(src_dir.rglob("*.result.json")):
    # Filenames are `<spec>__<impl>.result.json`.
    impl = path.name.removesuffix(".result.json").rsplit("__", 1)[-1]
    result = tokamax.AutotuningResult.loads(path.read_text())
    by_impl[impl] = result if impl not in by_impl else by_impl[impl] | result
    print(f"loaded {path.name} -> impl={impl}")

  if not by_impl:
    print(f"no *.result.json under {src_dir}", file=sys.stderr)
    return 1

  for impl, result in by_impl.items():
    path = cache_dir / cache_filename(mla_api.IMPLEMENTATIONS[impl])
    path.write_text(abstractify_result(result).dump_cache_str())
    print(f"wrote {path}")
  print(f"\ninstall with:  cp {cache_dir}/*.json tokamax/data/autotuning/tpu7x/")
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--spec", action="append", choices=SPEC_NAMES)
  parser.add_argument("--all", action="store_true")
  parser.add_argument("--impl", action="append", choices=IMPLS)
  parser.add_argument("--timeout", type=float, default=900.0)
  parser.add_argument("--max-workers", type=int, default=None)
  parser.add_argument("--out-dir", default=None)
  parser.add_argument(
      "--rebuild-cache-from",
      default=None,
      help=(
          "Directory of *.result.json files from a previous run. Regenerates"
          " the autotuning cache files from them and exits. Runs on CPU -- no"
          " TPU needed, so a failed dump never costs another sweep."
      ),
  )
  opts = parser.parse_args()

  if opts.rebuild_cache_from:
    return rebuild_cache(
        pathlib.Path(opts.rebuild_cache_from),
        pathlib.Path(opts.out_dir or opts.rebuild_cache_from),
    )

  if opts.all:
    spec_names = list(SPEC_NAMES)
  elif opts.spec:
    spec_names = opts.spec
  else:
    parser.error("pass --spec NAME (repeatable) or --all")
  impls = opts.impl or list(IMPLS)

  # The recipe copies `${REPO_DIR}/_cdk_out` into `$CDK_OUTPUT_DIR` after the
  # run, so that is the default even when the GCS mount is present.
  out_dir = pathlib.Path(
      opts.out_dir or os.environ.get("CDK_OUTPUT_DIR") or "_cdk_out"
  )
  out_dir.mkdir(parents=True, exist_ok=True)
  cache_dir = out_dir / "autotuning_cache"
  cache_dir.mkdir(parents=True, exist_ok=True)

  devices = jax.devices()
  print(f"jax {jax.__version__}, {len(devices)} devices: {devices[:8]}")
  if not devices or devices[0].platform != "tpu":
    print("FATAL: autotuning requires a TPU", file=sys.stderr)
    return 1
  device_kind = devices[0].device_kind
  print(f"device_kind: {device_kind}")
  print(f"output dir:  {out_dir.resolve()}")

  by_name = {s.name: s for s in mla_specs.ARG_SPECS}
  summaries = []
  merged = {}
  exit_code = 0

  for spec_name in spec_names:
    for impl_name in impls:
      try:
        summary = tune_one(
            by_name[spec_name],
            impl_name,
            out_dir,
            timeout=opts.timeout,
            max_workers=opts.max_workers,
        )
      except Exception:  # pylint: disable=broad-except
        # One bad pairing should not cost the whole sweep; the TPU hold is the
        # expensive part.
        print(f"\n!!! {spec_name}/{impl_name} raised:", file=sys.stderr)
        traceback.print_exc()
        summaries.append(
            dict(spec=spec_name, impl=impl_name, error=traceback.format_exc(limit=1))
        )
        exit_code = 1
        continue

      result = summary.pop("result")
      merged[impl_name] = (
          result if impl_name not in merged else merged[impl_name] | result
      )
      summaries.append(summary)

  # One cache file per op, matching `tokamax/data/autotuning/<device_kind>/`.
  # Wrapped: a serialization bug must not discard a sweep that already cost
  # ~20 minutes of TPU per (spec, impl). The per-pair `.result.json` files are
  # already on disk at this point, and `--rebuild-cache-from` can regenerate
  # the cache from them on CPU.
  for impl_name, result in merged.items():
    path = cache_dir / cache_filename(mla_api.IMPLEMENTATIONS[impl_name])
    try:
      with open(path, "w") as fh:
        fh.write(abstractify_result(result).dump_cache_str())
      print(f"\nwrote cache file {path}")
    except Exception:  # pylint: disable=broad-except
      print(f"\n!!! failed to write cache file {path}:", file=sys.stderr)
      traceback.print_exc()
      print(
          "    tuning data is preserved in the .result.json files; rerun with"
          " --rebuild-cache-from to regenerate",
          file=sys.stderr,
      )
      exit_code = 1

  with open(out_dir / "autotune_summary.json", "w") as fh:
    json.dump(summaries, fh, indent=2, default=str)

  print(f"\n{'=' * 70}")
  print("=== SUMMARY")
  print(f"{'=' * 70}")
  for s in summaries:
    if "error" in s:
      print(f"{s['spec']:28s} {s['impl']:3s}  RAISED")
      continue
    best = f"{s['best_median_ms']:.4f} ms" if s["best_median_ms"] else "n/a"
    print(
        f"{s['spec']:28s} {s['impl']:3s}  ok={s['ok']:3d} fail={s['failed']:3d}"
        f" (oom={s['oom']:3d})  best={best:>12s}  {s['wall_s']}s"
    )
  return exit_code


if __name__ == "__main__":
  sys.exit(main())

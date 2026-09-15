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

"""Summarise device time from `*.xplane.pb` traces, with no dependencies.

Why this exists: comparing MLA v3 kernel variants needs *device* time, and
untraced wall-clock has twice pointed the opposite way to the device numbers
(`disable_bounds_checks` read as +0.2% on wall-clock and -6.3% on device).
The usual readers -- `tensorflow.core.profiler`, `xprof`,
`tensorboard_plugin_profile` -- are none of them installed on the cdk VM or in
the job image, and the xprof UI is not reachable from a shell. So this walks
the protobuf wire format directly. XPlane's schema is small and stable enough
that a ~100-line reader is cheaper than a dependency.

Two numbers are reported per plane, and the difference between them matters:

  * **op total** -- the sum of event durations. Double-counts whenever ops
    overlap, which on TPU they routinely do (DMA against compute).
  * **occupancy** -- the measure of the *union* of the event intervals, i.e.
    the wall time during which the core was doing anything at all. This is the
    figure to compare across variants; it is what "device total" means
    elsewhere in this work.

Usage:
  python xplane_summary.py <dir-or-file> [<dir-or-file> ...] [--top N] [--json]

Directories are searched recursively for `*.xplane.pb`. Each trace is labelled
with the name of the directory `run_mla_profile.py` wrote it under, which
encodes the spec, impl and config, so output is self-describing.
"""

import argparse
import collections
import json
import pathlib
import sys

# ---------------------------------------------------------------------------
# Minimal protobuf wire-format reader.
#
# Only the four wire types that appear in xplane.proto are handled. Unknown
# fields are skipped rather than erroring, so a schema addition upstream
# degrades to missing data rather than a crash.
# ---------------------------------------------------------------------------

_VARINT, _I64, _LEN, _SGROUP, _EGROUP, _I32 = 0, 1, 2, 3, 4, 5


def _read_varint(buf, i):
  shift = result = 0
  while True:
    b = buf[i]
    i += 1
    result |= (b & 0x7F) << shift
    if not b & 0x80:
      return result, i
    shift += 7


def _fields(buf, start=0, end=None):
  """Yields `(field_number, wire_type, value)` for one message.

  `value` is an int for varint/fixed types and a `memoryview` slice for
  length-delimited ones -- slicing rather than copying keeps this usable on
  the ~100 MB traces a 10-iteration decode run produces.
  """
  i, end = start, len(buf) if end is None else end
  while i < end:
    key, i = _read_varint(buf, i)
    fn, wt = key >> 3, key & 7
    if wt == _VARINT:
      val, i = _read_varint(buf, i)
    elif wt == _I64:
      val, i = int.from_bytes(buf[i : i + 8], "little"), i + 8
    elif wt == _I32:
      val, i = int.from_bytes(buf[i : i + 4], "little"), i + 4
    elif wt == _LEN:
      n, i = _read_varint(buf, i)
      val, i = buf[i : i + n], i + n
    else:
      # Groups are deprecated and absent from xplane.proto; bail loudly rather
      # than silently mis-parsing the rest of the message.
      raise ValueError(f"unsupported wire type {wt} for field {fn}")
    yield fn, wt, val


def _first(buf, want, default=None):
  for fn, _, val in _fields(buf):
    if fn == want:
      return val
  return default


# ---------------------------------------------------------------------------
# xplane.proto accessors. Field numbers per
# tensorflow/core/profiler/protobuf/xplane.proto.
# ---------------------------------------------------------------------------


def _parse_plane(buf):
  """Returns `(name, {metadata_id: op_name}, [(start_ps, dur_ps, meta_id)])`."""
  name = ""
  meta = {}
  events = []
  for fn, _, val in _fields(buf):
    if fn == 2:  # string name
      name = bytes(val).decode("utf-8", "replace")
    elif fn == 4:  # map<int64, XEventMetadata> event_metadata
      entry_val = _first(val, 2)  # map value
      if entry_val is None:
        continue
      mid = _first(entry_val, 1, 0)  # XEventMetadata.id
      mname = _first(entry_val, 2)  # XEventMetadata.name
      if mname is not None:
        meta[mid] = bytes(mname).decode("utf-8", "replace")
    elif fn == 3:  # repeated XLine
      line_ts_ns = 0
      line_events = []
      for lfn, _, lval in _fields(val):
        if lfn == 3:  # XLine.timestamp_ns
          line_ts_ns = lval
        elif lfn == 4:  # repeated XEvent
          mid = off = dur = 0
          for efn, _, eval_ in _fields(lval):
            if efn == 1:
              mid = eval_
            elif efn == 2:
              off = eval_
            elif efn == 3:
              dur = eval_
          line_events.append((off, dur, mid))
      base_ps = line_ts_ns * 1000
      events.extend((base_ps + o, d, m) for o, d, m in line_events)
  return name, meta, events


def _occupancy_ps(events):
  """Measure of the union of `[start, start+dur)`, i.e. any-op-running time."""
  if not events:
    return 0
  ivals = sorted((s, s + d) for s, d, _ in events if d > 0)
  if not ivals:
    return 0
  total = 0
  cur_s, cur_e = ivals[0]
  for s, e in ivals[1:]:
    if s > cur_e:
      total += cur_e - cur_s
      cur_s, cur_e = s, e
    else:
      cur_e = max(cur_e, e)
  return total + cur_e - cur_s


def summarise(path, top=8):
  """Returns a list of per-plane summaries for one `.xplane.pb`."""
  buf = memoryview(pathlib.Path(path).read_bytes())
  out = []
  for fn, _, val in _fields(buf):
    if fn != 1:  # XSpace.planes
      continue
    name, meta, events = _parse_plane(val)
    # Host-side planes ("/host:CPU", "Host Threads", python tracing) are noise
    # for kernel comparison; keep only device cores.
    if "TPU" not in name and "device" not in name.lower():
      continue
    if not events:
      continue
    per_op = collections.Counter()
    for _, d, m in events:
      per_op[meta.get(m, f"<meta {m}>")] += d
    out.append({
        "plane": name,
        "n_events": len(events),
        "op_total_ms": sum(per_op.values()) / 1e9,
        "occupancy_ms": _occupancy_ps(events) / 1e9,
        "top_ops": [(k, v / 1e9) for k, v in per_op.most_common(top)],
    })
  return out


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("paths", nargs="+")
  ap.add_argument("--top", type=int, default=8)
  ap.add_argument("--json", action="store_true")
  ap.add_argument(
      "--iterations",
      type=int,
      default=1,
      help="Divide totals by this to get per-iteration time.",
  )
  args = ap.parse_args(argv)

  files = []
  for p in args.paths:
    p = pathlib.Path(p)
    files.extend(sorted(p.rglob("*.xplane.pb")) if p.is_dir() else [p])
  if not files:
    print("no *.xplane.pb found", file=sys.stderr)
    return 1

  results = []
  for f in files:
    # run_mla_profile.py writes each trace under a directory named for the
    # spec/impl/config, so that is the useful label -- not the .pb's own name,
    # which is just the worker id.
    label = f.parent
    while label.name in ("plugins", "profile") or label.name[:4].isdigit():
      label = label.parent
    try:
      planes = summarise(f, top=args.top)
    except Exception as e:  # pylint: disable=broad-except
      print(f"!! {label.name}: parse failed: {type(e).__name__}: {e}",
            file=sys.stderr)
      continue
    results.append({"label": label.name, "file": str(f), "planes": planes})

  if args.json:
    print(json.dumps(results, indent=2))
    return 0

  n = max(1, args.iterations)
  for r in results:
    print("=" * 78)
    print(r["label"])
    for pl in r["planes"]:
      print(f"  plane {pl['plane']}  ({pl['n_events']} events)")
      print(f"    occupancy  : {pl['occupancy_ms'] / n:9.4f} ms/iter"
            f"   <-- compare this across variants")
      print(f"    op total   : {pl['op_total_ms'] / n:9.4f} ms/iter"
            f"   (sums overlapping ops)")
      for name, ms in pl["top_ops"]:
        print(f"      {ms / n:9.4f} ms  {name}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

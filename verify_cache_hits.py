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

"""Checks that the installed autotuning cache is actually found at lookup.

A cache file whose keys do not match what a real call produces is not an
error -- `AutotuningCache` falls back to heuristics silently. So writing the
file proves nothing; this asserts the config a plain call resolves to is the
swept one rather than the heuristic.

For each spec, resolves the config with no explicit `config=` and compares it
to what `record_best_configs.py` wrote. Reports HIT, MISS (heuristic used) or
WRONG (a config was found but is not the recorded one).
"""

import sys

from tokamax._src.ops.experimental.mla import api as mla_api
from tokamax._src.ops.experimental.mla import arg_specs as mla_specs

from record_best_configs import BEST
from run_mla_autotune import materialize


def _resolved_config(op, args):
  """The config `Op.__call__` would use, cache included, without an override.

  `BoundArguments.get_config` is the real resolution order -- explicit config,
  then heuristics-if-null, then the autotuning cache, then heuristics. Calling
  it with `autotune_configs=None` means a cache miss silently falls through to
  heuristics, which is exactly the failure mode being tested for: an
  ill-keyed cache file produces no error, just the wrong config.
  """
  # Bare defaults are already the call path: check_autotuning_cache=True,
  # autotune_configs=None (so a miss does not silently trigger a sweep),
  # allow_heuristics=True. Passing them explicitly tripped the parameter
  # type-checker, so don't.
  return op.bind(**args).get_config()


def main():
  by_name = {s.name: s for s in mla_specs.ARG_SPECS}
  hits = misses = wrong = 0

  for spec_name, impl_name, overrides in BEST:
    spec = by_name[spec_name]
    op = mla_api.IMPLEMENTATIONS[impl_name]
    expected = op.config_cls(vmem_limit_bytes=64 * 1024 * 1024, **overrides)
    heuristic = op._get_heuristics_config(op.bind(**materialize(spec.args)))  # pylint: disable=protected-access

    try:
      got = _resolved_config(op, materialize(spec.args))
    except Exception as exc:  # pylint: disable=broad-except
      print(f"  {spec_name:38s} {impl_name:10s} RAISED "
            f"{type(exc).__name__}")
      if wrong == 0:  # full detail once, else the log is unreadable
        import traceback
        traceback.print_exc()
        print("FULLMSG:", str(exc)[:1500])
      wrong += 1
      continue

    if got == expected:
      verdict, n = "HIT  ", 0
      hits += 1
    elif got is None or got == heuristic:
      verdict, n = "MISS ", 1
      misses += 1
    else:
      verdict, n = "WRONG", 2
      wrong += 1
    del n
    print(f"  {spec_name:38s} {impl_name:10s} {verdict}")
    if verdict != "HIT  ":
      print(f"      expected: {expected}")
      print(f"      got     : {got}")

  print(f"\n{hits} hit, {misses} miss, {wrong} wrong")
  return 0 if (misses == 0 and wrong == 0) else 1


if __name__ == "__main__":
  sys.exit(main())

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

"""Emits the v2-vs-v3 MLA sweep results as CSV.

Every number here is transcribed from `mla_v2_v3_bench_test.py`, which is the
recorded artifact of the sweeps -- not re-measured by this script. Each row
carries its `provenance` field verbatim so a reader can tell which sweeps are
trustworthy.

Three things to know before reading the numbers:

  1. **All times are device time, not wallclock.** Wallclock inverted the
     result on four of the seven specs -- most severely on
     `chunked_prefill_f8_kv1024`, where 0.42 ms wall against 0.17 ms device
     means more than half the measurement was host overhead. Decisions were
     made on device occupancy (disjoint interval union) throughout.

  2. **Both sides are tuned.** v2 is swept as well as v3, so these are
     best-vs-best. An untuned-v2 comparison would flatter v3 and was
     explicitly rejected.

  3. **`total` includes v3's schedule-generation kernel**, which v2 has no
     equivalent of and which cannot overlap the main kernel because the main
     kernel consumes its output. That is why v3 can lose on `kernel` and still
     win on `total`.

All rows are measured on the current code, so every config shown can actually
be set. `prefill_bf16` reads 3.403x rather than the 3.469x recorded earlier:
that config used gate_stitch=False, which no longer exists, and its +3.2% on
the v3 kernel is exactly that flag's predicted cost.

xprof: no traces are currently on disk (`/mnt/nfs/ajaygopi/xp` is empty).
`run_final_traces.sh` regenerates them via `run_mla_profile.py`, retrieved
with `cdk job xprof <id> -a`; that needs a TPU job.

Sequence lengths are two plain integer columns, `q_len` and `kv_len`, not a
"(q,kv)" string. Spreadsheets read a parenthesised value as accounting
notation for a negative number, so "(1,8192)" opens as -18192.

Usage:
  python make_results_csv.py                 # writes mla_v2_v3_results.csv
  python make_results_csv.py --stdout
  python make_results_csv.py --xlsx out.xlsx # two sheets: Results, Parameters
"""

import argparse
import csv
import io
import sys

# spec -> (q_len, kv_len, num_seqs, page_size, dtype, provenance)
_SPECS = {
    "decode_bf16": (1, 8192, 3, 256, "bf16", "TUNED_DONATED"),
    "decode_f8": (1, 8192, 128, 256, "fp8", "TUNED_DONATED"),
    "decode_f8_kv9216": (1, 9216, 128, 256, "fp8", "TUNED_DONATED"),
    "decode_f8_kv9216_p1024": (1, 9216, 128, 1024, "fp8", "TUNED_DONATED"),
    "prefill_bf16": (2048, 2048, 2, 256, "bf16", "TUNED_DONATED"),
    "chunked_prefill_f8_kv1024": (256, 1024, 1, 256, "fp8",
                                  "TUNED_DONATED"),
    "chunked_prefill_f8_kv8192": (256, 8192, 1, 256, "fp8",
                                  "TUNED_DONATED"),
}

# spec -> impl -> (config dict, total_ms, kernel_ms)
#
# MEASURED, not transcribed: device occupancy from the xprof traces in cdk
# job j-c99bed00-97ec-4982-af5f (2026-09-15), per-iteration over 10
# iterations, on the current code with `bands` removed. Every config below
# is one that can actually be set today.
_RESULTS = {
    "decode_bf16": {
        "v2": (dict(num_kv_pages_per_block=8, num_queries_per_block=1,
                    decode_batch_size=1, mixed_q_split=1), 0.0533, 0.028),
        # The whole loss here is dispatch, not the kernel: v3's kernel is still
        # 1.019x faster (0.0274 vs 0.0280), but with `bands` gone it pays three
        # lax.conds where it used to run one pass straight through, and this is
        # the smallest workload in the set (3 sequences) so fixed per-call cost
        # dominates. 0.0529 -> 0.0728 total.
        "v3": (dict(num_kv_pages_per_block=8, num_queries_per_block=1,
                    batch_size=4, n_buffer=2), 0.0728, 0.0274),
    },
    "decode_f8": {
        "v2": (dict(num_kv_pages_per_block=8, num_queries_per_block=1,
                    decode_batch_size=8, mixed_q_split=1), 0.48, 0.4164),
        "v3": (dict(num_kv_pages_per_block=32, num_queries_per_block=1,
                    batch_size=2, n_buffer=2), 0.472, 0.4112),
    },
    "decode_f8_kv9216": {
        "v2": (dict(num_kv_pages_per_block=12, num_queries_per_block=1,
                    decode_batch_size=8, mixed_q_split=1), 0.4896, 0.4265),
        "v3": (dict(num_kv_pages_per_block=36, num_queries_per_block=1,
                    batch_size=2, n_buffer=2), 0.5179, 0.4537),
    },
    "decode_f8_kv9216_p1024": {
        "v2": (dict(num_kv_pages_per_block=3, num_queries_per_block=1,
                    decode_batch_size=8, mixed_q_split=1), 0.4652, 0.4023),
        "v3": (dict(num_kv_pages_per_block=3, num_queries_per_block=1,
                    batch_size=4, n_buffer=2), 0.5063, 0.4267),
    },
    "prefill_bf16": {
        "v2": (dict(num_kv_pages_per_block=8, num_queries_per_block=16,
                    decode_batch_size=1, mixed_q_split=4), 9.7253, 8.0844),
        # Was 2.7806/2.6138 (3.469x) under gate_stitch=False and bands=mixed,
        # neither of which exists now.
        "v3": (dict(num_kv_pages_per_block=2, num_queries_per_block=32,
                    batch_size=1, n_buffer=2, q_split=8,
                    kv_slack_pad_lanes=0), 2.8922, 2.6887),
    },
    "chunked_prefill_f8_kv1024": {
        "v2": (dict(num_kv_pages_per_block=4, num_queries_per_block=32,
                    decode_batch_size=1, mixed_q_split=4), 0.1756, 0.0941),
        "v3": (dict(num_kv_pages_per_block=4, num_queries_per_block=32,
                    batch_size=1, n_buffer=2, q_split=8,
                    p_same_dtype_as_v=True), 0.1332, 0.0987),
    },
    "chunked_prefill_f8_kv8192": {
        "v2": (dict(num_kv_pages_per_block=8, num_queries_per_block=32,
                    decode_batch_size=1, mixed_q_split=4), 0.6106, 0.529),
        "v3": (dict(num_kv_pages_per_block=8, num_queries_per_block=32,
                    batch_size=1, n_buffer=2, q_split=16,
                    p_same_dtype_as_v=True), 0.6269, 0.5719),
    },
}

# Everything is measured on current code, so nothing is unreachable now.
_UNREACHABLE = {}

# xprof traces from cdk job j-c99bed00-97ec-4982-af5f (2026-09-15), the run
# that also returned mla_kernel_v3_test 87/87 with `bands` removed. Keyed
# "<spec>|<impl>"; paste into go/cdk-xprof. v3 rows traced v3_native against
# the _v3layout spec, so the KV-layout conversion is outside the timed region.
_XPROF = {
    "chunked_prefill_f8_kv1024|v2": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/chunked_prefill_f8_kv1024__v2__num_kv_pages_per_block4_num_queries_per_block32_decode_batch_size1_mixed_q_split4_vmem_limit_bytes67108864/plugins/profile/2026_09_15_18_11_27/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "chunked_prefill_f8_kv1024|v3": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/chunked_prefill_f8_kv1024_v3layout__v3_native__num_kv_pages_per_block4_num_queries_per_block32_batch_size1_q_split8_p_same_dtype_as_vTrue_vmem_limit_bytes67108864_n_buffer2/plugins/profile/2026_09_15_18_11_45/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "chunked_prefill_f8_kv8192|v2": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/chunked_prefill_f8_kv8192__v2__num_kv_pages_per_block8_num_queries_per_block32_decode_batch_size1_mixed_q_split4_vmem_limit_bytes67108864/plugins/profile/2026_09_15_18_11_57/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "chunked_prefill_f8_kv8192|v3": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/chunked_prefill_f8_kv8192_v3layout__v3_native__num_kv_pages_per_block8_num_queries_per_block32_batch_size1_q_split16_p_same_dtype_as_vTrue_vmem_limit_bytes67108864_n_buffer2/plugins/profile/2026_09_15_18_12_19/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "decode_bf16|v2": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/decode_bf16__v2__num_kv_pages_per_block8_num_queries_per_block1_decode_batch_size1_mixed_q_split1_vmem_limit_bytes67108864/plugins/profile/2026_09_15_18_08_30/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "decode_bf16|v3": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/decode_bf16_v3layout__v3_native__num_kv_pages_per_block8_num_queries_per_block1_batch_size4_vmem_limit_bytes67108864_n_buffer2/plugins/profile/2026_09_15_18_08_52/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "decode_f8|v2": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/decode_f8__v2__num_kv_pages_per_block8_num_queries_per_block1_decode_batch_size8_mixed_q_split1_vmem_limit_bytes67108864/plugins/profile/2026_09_15_18_09_05/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "decode_f8_kv9216|v2": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/decode_f8_kv9216__v2__num_kv_pages_per_block12_num_queries_per_block1_decode_batch_size8_mixed_q_split1_vmem_limit_bytes67108864/plugins/profile/2026_09_15_18_09_46/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "decode_f8_kv9216_p1024|v2": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/decode_f8_kv9216_p1024__v2__num_kv_pages_per_block3_num_queries_per_block1_decode_batch_size8_mixed_q_split1_vmem_limit_bytes67108864/plugins/profile/2026_09_15_18_10_28/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "decode_f8_kv9216_p1024|v3": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/decode_f8_kv9216_p1024_v3layout__v3_native__num_kv_pages_per_block3_num_queries_per_block1_batch_size4_vmem_limit_bytes67108864_n_buffer2/plugins/profile/2026_09_15_18_10_47/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "decode_f8_kv9216|v3": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/decode_f8_kv9216_v3layout__v3_native__num_kv_pages_per_block36_num_queries_per_block1_batch_size2_vmem_limit_bytes67108864_n_buffer2/plugins/profile/2026_09_15_18_10_15/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "decode_f8|v3": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/decode_f8_v3layout__v3_native__num_kv_pages_per_block32_num_queries_per_block1_batch_size2_vmem_limit_bytes67108864_n_buffer2/plugins/profile/2026_09_15_18_09_32/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "prefill_bf16|v2": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/prefill_bf16__v2__num_kv_pages_per_block8_num_queries_per_block16_decode_batch_size1_mixed_q_split4_vmem_limit_bytes67108864/plugins/profile/2026_09_15_18_11_00/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb",
    "prefill_bf16|v3": "go/cdk-xprof?gs://cloud-devkit/jobs/j-c99bed00-97ec-4982-af5f/outputs/j-c99bed00-97ec-4982-af5f-tokamax-0/j-c99bed00-97ec-4982-af5f-tokamax-0-0-gxbml/tokamax/xprof/prefill_bf16_v3layout__v3_native__num_kv_pages_per_block2_num_queries_per_block32_batch_size1_q_split8_kv_slack_pad_lanes0_vmem_limit_bytes67108864_n_buffer2/plugins/profile/2026_09_15_18_11_16/j-c99bed00-97ec-4982-af5f-tokamax-0-0.xplane.pb"
}

_FIELDS = [
    "spec", "q_len", "kv_len", "num_seqs", "page_size", "dtype", "provenance",
    "impl",
    # block sizes, both implementations
    "num_kv_pages_per_block", "num_queries_per_block", "bkv_sz", "bq_sz",
    # v2-only
    "decode_batch_size", "mixed_q_split",
    # v3-only
    "batch_size", "n_buffer", "q_split", "bq_c_sz", "bands",
    "kv_slack_pad_lanes", "p_same_dtype_as_v",
    # timings, device
    "device_total_ms", "device_kernel_ms", "device_non_kernel_ms",
    # comparisons (populated on the v3 row)
    "total_speedup_v3_over_v2", "kernel_speedup_v3_over_v2", "winner",
    "reachable_now", "note",
]


def _rows():
  for spec, (q_len, kv_len, nseq, page, dtype, prov) in _SPECS.items():
    v2_cfg, v2_tot, v2_ker = _RESULTS[spec]["v2"]
    v3_cfg, v3_tot, v3_ker = _RESULTS[spec]["v3"]
    for impl, (cfg, tot, ker) in (
        ("v2", (v2_cfg, v2_tot, v2_ker)),
        ("v3", (v3_cfg, v3_tot, v3_ker)),
    ):
      bq = cfg["num_queries_per_block"]
      qsplit = cfg.get("q_split", 1)
      row = {
          "spec": spec, "q_len": q_len, "kv_len": kv_len, "num_seqs": nseq,
          "page_size": page, "dtype": dtype, "provenance": prov, "impl": impl,
          "num_kv_pages_per_block": cfg["num_kv_pages_per_block"],
          "num_queries_per_block": bq,
          "bkv_sz": cfg["num_kv_pages_per_block"] * page,
          "bq_sz": bq,
          "decode_batch_size": cfg.get("decode_batch_size", ""),
          "mixed_q_split": cfg.get("mixed_q_split", ""),
          "batch_size": cfg.get("batch_size", ""),
          "n_buffer": cfg.get("n_buffer", ""),
          "q_split": qsplit if impl == "v3" else "",
          "bq_c_sz": (bq // qsplit) if impl == "v3" else "",
          "kv_slack_pad_lanes": (
              cfg.get("kv_slack_pad_lanes", 128) if impl == "v3" else ""),
          "p_same_dtype_as_v": (
              cfg.get("p_same_dtype_as_v", False) if impl == "v3" else ""),
          "device_total_ms": f"{tot:.4f}",
          "device_kernel_ms": f"{ker:.4f}",
          "device_non_kernel_ms": f"{tot - ker:.4f}",
          "total_speedup_v3_over_v2": "",
          "kernel_speedup_v3_over_v2": "",
          "winner": "",
          "reachable_now": "yes",
          "note": "",
      }
      if impl == "v3":
        row["total_speedup_v3_over_v2"] = f"{v2_tot / v3_tot:.3f}"
        row["kernel_speedup_v3_over_v2"] = f"{v2_ker / v3_ker:.3f}"
        row["winner"] = "v3" if v3_tot < v2_tot else "v2"
      if (spec, impl) in _UNREACHABLE:
        row["reachable_now"] = "no"
        row["note"] = _UNREACHABLE[(spec, impl)]
      yield row


_RESULTS_HDR = [
    "spec", "q_len", "kv_len", "num_seqs", "page_size", "dtype",
    "v2_total_ms", "v3_total_ms", "total_speedup_v3_over_v2",
    "v2_kernel_ms", "v3_kernel_ms", "kernel_speedup_v3_over_v2",
    "v2_non_kernel_ms", "v3_non_kernel_ms", "winner", "note",
    "v2_xprof", "v3_xprof",
]

_PARAMS_HDR = [
    "spec", "impl", "page_size",
    "num_kv_pages_per_block", "bkv_sz", "num_queries_per_block", "bq_sz",
    "q_split", "bq_c_sz", "batch_size", "n_buffer",
    "kv_slack_pad_lanes", "p_same_dtype_as_v",
    "decode_batch_size", "mixed_q_split", "reachable_now", "note",
]


def _sheet_rows():
  """(results_rows, params_rows) -- one summary row per spec, params per impl."""
  results, params = [], []
  by_spec = {}
  for r in _rows():
    by_spec.setdefault(r["spec"], {})[r["impl"]] = r
    params.append([
        r["spec"], r["impl"], int(r["page_size"]),
        r["num_kv_pages_per_block"], r["bkv_sz"],
        r["num_queries_per_block"], r["bq_sz"],
        r["q_split"], r["bq_c_sz"], r["batch_size"], r["n_buffer"],
        r["kv_slack_pad_lanes"], r["p_same_dtype_as_v"],
        r["decode_batch_size"], r["mixed_q_split"],
        r["reachable_now"], r["note"],
    ])
  for spec, d in by_spec.items():
    v2, v3 = d["v2"], d["v3"]
    results.append([
        spec, int(v2["q_len"]), int(v2["kv_len"]), int(v2["num_seqs"]),
        int(v2["page_size"]), v2["dtype"],
        float(v2["device_total_ms"]), float(v3["device_total_ms"]),
        float(v3["total_speedup_v3_over_v2"]),
        float(v2["device_kernel_ms"]), float(v3["device_kernel_ms"]),
        float(v3["kernel_speedup_v3_over_v2"]),
        float(v2["device_non_kernel_ms"]), float(v3["device_non_kernel_ms"]),
        v3["winner"], v3["note"],
        _XPROF.get(f"{spec}|v2", ""), _XPROF.get(f"{spec}|v3", ""),
    ])
  results.sort(key=lambda r: -r[8])  # fastest v3 speedup first
  return results, params


def _write_xlsx(path):
  """Two sheets: a per-spec summary, and the config chosen for each impl."""
  from openpyxl import Workbook  # pylint: disable=g-import-not-at-top
  from openpyxl.styles import Alignment, Font, PatternFill  # pylint: disable=g-import-not-at-top
  from openpyxl.utils import get_column_letter  # pylint: disable=g-import-not-at-top

  results, params = _sheet_rows()
  wb = Workbook()
  head_font = Font(bold=True, color="FFFFFF")
  head_fill = PatternFill("solid", fgColor="44546A")
  win = PatternFill("solid", fgColor="E2EFDA")
  lose = PatternFill("solid", fgColor="FCE4E4")

  def style(ws, hdr, rows, numfmt=None):
    ws.append(hdr)
    for c in ws[1]:
      c.font, c.fill = head_font, head_fill
      c.alignment = Alignment(horizontal="center", wrap_text=True)
    for r in rows:
      ws.append(r)
    ws.freeze_panes = "A2"
    for i, name in enumerate(hdr, start=1):
      width = max(len(str(name)), *(len(str(r[i - 1])) for r in rows)) + 2
      ws.column_dimensions[get_column_letter(i)].width = min(width, 26)
      if numfmt and name in numfmt:
        for row in range(2, len(rows) + 2):
          ws.cell(row=row, column=i).number_format = numfmt[name]

  ws = wb.active
  ws.title = "Results"
  style(ws, _RESULTS_HDR, results, numfmt={
      "v2_total_ms": "0.0000", "v3_total_ms": "0.0000",
      "v2_kernel_ms": "0.0000", "v3_kernel_ms": "0.0000",
      "v2_non_kernel_ms": "0.0000", "v3_non_kernel_ms": "0.0000",
      "total_speedup_v3_over_v2": '0.000"x"',
      "kernel_speedup_v3_over_v2": '0.000"x"',
  })
  # Tint the speedup columns by who won; >1 means v3 is faster.
  for row in range(2, len(results) + 2):
    for col in (9, 12):
      cell = ws.cell(row=row, column=col)
      cell.fill = win if (cell.value or 0) > 1.0 else lose

  style(wb.create_sheet("Parameters"), _PARAMS_HDR, params)
  wb.save(path)
  return len(results), len(params)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out", default="mla_v2_v3_results.csv")
  ap.add_argument("--stdout", action="store_true")
  ap.add_argument("--xlsx", help="also write a two-sheet workbook here")
  opts = ap.parse_args(argv)

  buf = io.StringIO()
  w = csv.DictWriter(buf, fieldnames=_FIELDS)
  w.writeheader()
  for r in _rows():
    w.writerow(r)

  if opts.stdout:
    sys.stdout.write(buf.getvalue())
  else:
    with open(opts.out, "w", newline="") as f:
      f.write(buf.getvalue())
    print(f"wrote {opts.out} ({len(list(_rows()))} rows)")
  if opts.xlsx:
    n_res, n_par = _write_xlsx(opts.xlsx)
    print(f"wrote {opts.xlsx} (Results: {n_res} rows, Parameters: {n_par})")
  return 0


if __name__ == "__main__":
  sys.exit(main())

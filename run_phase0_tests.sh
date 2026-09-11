#!/bin/bash
# Phase 0 validation for the MLA v2/v3 tokamax `Op` wrappers.
#
# Each test file runs in its own pytest process on purpose. `v2/mla_kernel_v2_test.py`
# and `v3/mla_kernel_v3_test.py` both call `flags.DEFINE_bool("debug_mode", ...)`
# at import time, so collecting them together raises
# `absl.flags._exceptions.DuplicateFlagError`. The recipe's default RUN_CMD has
# the same problem.
#
# Runs every suite even if an earlier one fails, so one command gives the whole
# picture, then exits non-zero if any failed.

set -u

declare -A RESULTS
FILES=(
  "tokamax/_src/ops/experimental/mla/v2_v3_op_test.py"
  "tokamax/_src/ops/experimental/mla/v3/mla_kernel_v3_test.py"
  "tokamax/_src/ops/experimental/mla/v2/mla_kernel_v2_test.py"
)

overall=0
for f in "${FILES[@]}"; do
  echo ""
  echo "############################################################"
  echo "### pytest ${f}"
  echo "############################################################"
  pytest -q -p no:cacheprovider "${f}"
  rc=$?
  RESULTS["${f}"]=${rc}
  [ ${rc} -ne 0 ] && overall=1
done

echo ""
echo "############################################################"
echo "### PHASE 0 SUMMARY"
echo "############################################################"
for f in "${FILES[@]}"; do
  if [ "${RESULTS[$f]}" -eq 0 ]; then status="PASS"; else status="FAIL"; fi
  printf '%-6s (rc=%s) %s\n' "${status}" "${RESULTS[$f]}" "${f}"
done
echo "### overall: ${overall}"
exit ${overall}

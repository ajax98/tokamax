#!/bin/bash
# TPU correctness suite for the MLA v2/v3 flag-removal work.
#
# Run as a cdk job via the `tokamax-mla-test` recipe:
#   cdk job create tokamax-mla-test RUN_CMD="bash run_tpu_correctness.sh" ...
#
# Two things force the structure here:
#
#  * v2's and v3's kernel tests both define an absl flag named `debug_mode`,
#    so collecting them in one pytest invocation dies with DuplicateFlagError
#    before a single test runs. Each file gets its own invocation.
#
#  * `mla_kernel_v3_test.py` can abort the interpreter outright ("Fatal Python
#    error: Aborted" inside a device copy), which takes the rest of the run
#    with it and loses pytest's failure summary. No pytest-forked here, so
#    that file is driven one test per process. Slower, but a crash then costs
#    one line of the report instead of everything after it.
#
# Exits non-zero if anything failed or crashed.

set -u
cd "${REPO_DIR:-/mnt/nfs/ajaygopi/my-tokamax-fork}" || exit 1
MLA=tokamax/_src/ops/experimental/mla
PYTEST="python -m pytest -q -p no:cacheprovider --no-header"

overall=0
last_rc=0
declare -a REPORT=()

run_whole() {  # $1 = path
  echo ""
  echo "################ ${1}"
  out=$(${PYTEST} -rf "${1}" 2>&1); rc=$?
  last_rc=${rc}
  echo "${out}"
  REPORT+=("$(printf '%-34s rc=%-3s %s' "$(basename "${1}")" "${rc}" \
    "$(echo "${out}" | grep -E '[0-9]+ (passed|failed|skipped|error)' | tail -1)")")
  [ "${rc}" -ne 0 ] && overall=1
  return 0
}

run_isolated() {  # $1 = path; one process per test id
  echo ""
  echo "################ ${1}  (one process per test)"
  # NB: no extra -q here. ${PYTEST} already has one, and -qq raises the quiet
  # level far enough to suppress the id lines entirely (collected 0).
  mapfile -t IDS < <(${PYTEST} --collect-only "${1}" 2>/dev/null \
                     | grep '::' | sed 's/[[:space:]]*$//')
  echo "collected ${#IDS[@]} test ids"
  local pass=0 fail=0 crash=0 skip=0
  for id in "${IDS[@]}"; do
    out=$(timeout 300 ${PYTEST} "${id}" 2>&1); rc=$?
    case "${rc}" in
      0) if echo "${out}" | grep -q 'skipped'; then skip=$((skip+1));
         else pass=$((pass+1)); fi ;;
      1) fail=$((fail+1)); echo "FAIL  ${id}"
         echo "${out}" | grep -E 'Error|assert|Mismatch|E  ' | head -8 ;;
      *) crash=$((crash+1)); echo "CRASH(rc=${rc})  ${id}"
         echo "${out}" | grep -E 'Fatal|Aborted|RESOURCE_EXHAUSTED|Check failed|VMEM|OOM' | head -8 ;;
    esac
  done
  echo "  -> pass=${pass} fail=${fail} crash=${crash} skip=${skip}"
  REPORT+=("$(printf '%-34s pass=%-4s fail=%-4s crash=%-4s skip=%s' \
    "$(basename "${1}")" "${pass}" "${fail}" "${crash}" "${skip}")")
  [ "${fail}" -ne 0 ] || [ "${crash}" -ne 0 ] && overall=1
  return 0
}

run_whole "${MLA}/v3/schedule_test.py"
run_whole "${MLA}/v3/flash_attention_test.py"
run_whole "${MLA}/v2_v3_op_test.py"
run_whole "${MLA}/v2/mla_kernel_v2_test.py"

# Try this one in a single process first -- 87 separate processes cost ~20
# minutes of TPU init. Only fall back to isolation if it fails or aborts, i.e.
# exactly when the per-test attribution is worth paying for.
run_whole "${MLA}/v3/mla_kernel_v3_test.py"
# Gate on *this* suite's exit code, not `overall` -- overall is already 1 from
# the known pre-existing VMEM OOM in v2_v3_op_test, which would make the
# fallback fire unconditionally.
if [ "${last_rc}" -ne 0 ]; then
  echo ""
  echo "### whole-file run of mla_kernel_v3_test was not clean; isolating"
  run_isolated "${MLA}/v3/mla_kernel_v3_test.py"
fi

echo ""
echo "################ SUMMARY"
for line in "${REPORT[@]}"; do echo "  ${line}"; done
echo "### overall exit ${overall}"
exit "${overall}"

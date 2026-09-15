#!/bin/bash
# xprof traces for all seven v2-vs-v3 comparisons, each side at its best config.
#
#   cdk job create tokamax-mla-test RUN_CMD="bash run_xprof_all.sh" ...
#   cdk job xprof <job-id> -a          # links
#   cdk job sync-outputs <job-id> ...  # raw *.xplane.pb
#
# Traces land in $CDK_OUTPUT_DIR/xprof/<spec>__<impl>/. Anywhere else and
# `cdk job xprof` cannot see them -- it scans the job's GCS outputs prefix.
#
# v3 runs against the `_v3layout` twin of each spec with impl `v3_native`,
# because v3_native consumes a SEQ_ALONG_LANE cache directly. Tracing plain
# `v3` instead would put the layout conversion inside the timed region and
# measure drop-in migration cost rather than steady-state kernel cost.
#
# Configs are the swept winners, minus every key that no longer exists:
# gate_stitch, two_step_flash_attention and bands are all gone. prefill_bf16
# was tuned with gate_stitch=False and the decode specs with bands="decode",
# so those shapes trace slower here than their historical best. That is
# current reality, which is the point of re-tracing.

set -u
cd "${REPO_DIR:-/mnt/nfs/ajaygopi/my-tokamax-fork}" || exit 1
ITERS="${ITERS:-10}"
rc_all=0

trace() {  # $1=spec  $2=impl  $3=variant
  echo ""
  echo "################ ${1}  /  ${2}"
  echo "################ ${3}"
  python run_mla_profile.py --spec "${1}" --impl "${2}" \
    --iterations "${ITERS}" --variant "${3}"
  local rc=$?
  [ "${rc}" -ne 0 ] && { echo "!!! ${1}/${2} exited ${rc}"; rc_all=1; }
  return 0
}

V2_COMMON="vmem_limit_bytes=67108864"
V3_COMMON="vmem_limit_bytes=67108864,n_buffer=2"

# ---- decode ---------------------------------------------------------------
trace decode_bf16 v2 \
  "num_kv_pages_per_block=8,num_queries_per_block=1,decode_batch_size=1,mixed_q_split=1,${V2_COMMON}"
trace decode_bf16_v3layout v3_native \
  "num_kv_pages_per_block=8,num_queries_per_block=1,batch_size=4,${V3_COMMON}"

trace decode_f8 v2 \
  "num_kv_pages_per_block=8,num_queries_per_block=1,decode_batch_size=8,mixed_q_split=1,${V2_COMMON}"
trace decode_f8_v3layout v3_native \
  "num_kv_pages_per_block=32,num_queries_per_block=1,batch_size=2,${V3_COMMON}"

trace decode_f8_kv9216 v2 \
  "num_kv_pages_per_block=12,num_queries_per_block=1,decode_batch_size=8,mixed_q_split=1,${V2_COMMON}"
trace decode_f8_kv9216_v3layout v3_native \
  "num_kv_pages_per_block=36,num_queries_per_block=1,batch_size=2,${V3_COMMON}"

trace decode_f8_kv9216_p1024 v2 \
  "num_kv_pages_per_block=3,num_queries_per_block=1,decode_batch_size=8,mixed_q_split=1,${V2_COMMON}"
trace decode_f8_kv9216_p1024_v3layout v3_native \
  "num_kv_pages_per_block=3,num_queries_per_block=1,batch_size=4,${V3_COMMON}"

# ---- prefill / chunked prefill -------------------------------------------
trace prefill_bf16 v2 \
  "num_kv_pages_per_block=8,num_queries_per_block=16,decode_batch_size=1,mixed_q_split=4,${V2_COMMON}"
trace prefill_bf16_v3layout v3_native \
  "num_kv_pages_per_block=2,num_queries_per_block=32,batch_size=1,q_split=8,kv_slack_pad_lanes=0,${V3_COMMON}"

trace chunked_prefill_f8_kv1024 v2 \
  "num_kv_pages_per_block=4,num_queries_per_block=32,decode_batch_size=1,mixed_q_split=4,${V2_COMMON}"
trace chunked_prefill_f8_kv1024_v3layout v3_native \
  "num_kv_pages_per_block=4,num_queries_per_block=32,batch_size=1,q_split=8,p_same_dtype_as_v=True,${V3_COMMON}"

trace chunked_prefill_f8_kv8192 v2 \
  "num_kv_pages_per_block=8,num_queries_per_block=32,decode_batch_size=1,mixed_q_split=4,${V2_COMMON}"
trace chunked_prefill_f8_kv8192_v3layout v3_native \
  "num_kv_pages_per_block=8,num_queries_per_block=32,batch_size=1,q_split=16,p_same_dtype_as_v=True,${V3_COMMON}"

echo ""
echo "################ xprof inventory"
find "${CDK_OUTPUT_DIR:-_cdk_out}/xprof" -name '*.xplane.pb' 2>/dev/null \
  | sed "s|.*/xprof/||" | sort
n=$(find "${CDK_OUTPUT_DIR:-_cdk_out}/xprof" -name '*.xplane.pb' 2>/dev/null | wc -l)
echo "### ${n} xplane files (expect 14, one per spec x impl)"
[ "${n}" -lt 14 ] && rc_all=1
echo "### overall exit ${rc_all}"
exit "${rc_all}"

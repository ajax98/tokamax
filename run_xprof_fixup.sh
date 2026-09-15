#!/bin/bash
# Re-trace the three specs whose configs in run_xprof_all.sh disagreed with
# mla_v2_v3_bench_test.py, which is the file that recorded the timings and so
# is the source of truth for what produced them.
#
#   kv8192 / kv1024  were missing p_same_dtype_as_v=True
#   kv9216_p1024     had bands=decode; the source sets kv_slack_pad_lanes=128
#                    and no bands
#
# Only the v3 side changed, so only v3 is re-traced; the v2 numbers from
# j-dc9cab58 stand.
set -u
cd "${REPO_DIR:-/mnt/nfs/ajaygopi/my-tokamax-fork}" || exit 1
C="vmem_limit_bytes=67108864,n_buffer=2"
rc=0
t() { echo ""; echo "######## $1"; python run_mla_profile.py --spec "$1" --impl v3_native \
        --iterations 10 --variant "$2" || rc=1; }

t chunked_prefill_f8_kv8192_v3layout \
  "num_kv_pages_per_block=8,num_queries_per_block=32,batch_size=1,q_split=16,bands=mixed,p_same_dtype_as_v=True,${C}"
t chunked_prefill_f8_kv1024_v3layout \
  "num_kv_pages_per_block=4,num_queries_per_block=32,batch_size=1,q_split=8,p_same_dtype_as_v=True,${C}"
t decode_f8_kv9216_p1024_v3layout \
  "num_kv_pages_per_block=3,num_queries_per_block=1,batch_size=4,kv_slack_pad_lanes=128,${C}"

n=$(find "${CDK_OUTPUT_DIR:-_cdk_out}/xprof" -name '*.xplane.pb' 2>/dev/null | wc -l)
echo "### ${n} xplane files (expect 3)"; [ "${n}" -lt 3 ] && rc=1
echo "### overall exit ${rc}"; exit "${rc}"

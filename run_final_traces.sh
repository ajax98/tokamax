#!/bin/bash
# Final side-by-side traces: v2 vs v3_native on decode_f8_kv9216 at
# page_size=1024, each at its own best-known config.
#
# Emits both trace kinds:
#   * xprof  -- device-level, written by run_mla_profile.py into $CDK_OUTPUT_DIR
#               as *.xplane.pb; retrieve with `cdk job xprof <id> -a`.
#   * Perfetto -- host-level, parsed by job_manager out of the container logs.
#               It only materialises when the TPU VLOGs below are on; without
#               them the job reports "Perfetto trace generated (0 events)" and
#               no URL appears. Retrieve from `cdk job list -o json` /
#               go/cdk-perfetto/<job-id>.
export TPU_VMODULE="tpu_pjrt_client=1,pjrt_stream_executor_client=1,tpu_pjrt_compiler_utils=1"
export TPU_STDERR_LOG_LEVEL=0

set -u
COMMON="--iterations 10"

echo "############ v2 (kv=3, q=1, decode_batch_size=8) ############"
python run_mla_profile.py --spec decode_f8_kv9216_p1024 --impl v2 ${COMMON} \
  --variant num_kv_pages_per_block=3,num_queries_per_block=1,decode_batch_size=8
rc_v2=$?

echo "############ v3_native (kv=3, q=1, b=4, n=2, + all landed wins) ############"
python run_mla_profile.py --spec decode_f8_kv9216_p1024_v3layout --impl v3_native ${COMMON} \
  --variant num_queries_per_block=1,num_kv_pages_per_block=3,batch_size=4,n_buffer=2,fast_mask=True,disable_bounds_checks=True,merge_kv_dma=True,kv_slack_pad_lanes=128
rc_v3=$?

echo "############ done: v2 rc=${rc_v2}  v3 rc=${rc_v3} ############"
exit $(( rc_v2 | rc_v3 ))

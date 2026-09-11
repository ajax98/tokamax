#!/bin/bash
# Hand grid-sweep of v3_native on decode_f8_kv9216, under the donated harness.
#
# Why by hand rather than `tokamax.autotune`: the autotuner ranks configs by a
# metric polluted by the non-donated 755 MB cache copy, and it demonstrably
# mis-ranked -- it picked num_kv_pages_per_block=1 (1.994 ms) over 8 (1.037 ms).
# `run_mla_profile.py` donates and chains the cache, so its numbers are the ones
# to trust until the same fix lands inside tokamax's benchmarking path.
#
# Axes
# ----
# num_queries_per_block {1,4,16}
#   Expected to be *inert at runtime* on this spec: distribution is
#   [128,128,128], so only the DECODE pass runs, and `calculate_block_sizes`
#   pins bq_sz=1 there regardless -- q only feeds the prefill block sizes.
#   (v2 does the same thing by a different route: bq_sz =
#   min(num_queries_per_block, static_q_len=1).)
#
#   It is swept anyway because it is *not* inert at compile time. The
#   PREFILL/MIXED branches are compiled even though they never execute, and
#   `vmem_limit_bytes` is checked per pallas_call, so a large q can fail
#   compilation in a branch that would never have run. Everything scaling with
#   bq_sz -- acc_scratch, m/l scratch, and the Q/O staging buffers (times
#   batch_size times n_buffer) -- costs ~11 MB more at q=16 than q=4. So q=1
#   should let strictly more of the grid survive, and if the runtime numbers
#   come back equal across q, that confirms the inertness claim and makes q=1
#   the right decode default.
#
# num_kv_pages_per_block {4,8,16,32}
#   1 and 2 already measured at ~1.96-1.99 ms, clearly off. 32 added to match
#   v2's grid.
#
# batch_size {1,2,3,4,8}
#   3 included: nothing requires a power of two. `flush_to_hbm` does
#   `align_to(count, batch_size)` and masks the remainder.
#
# n_buffer {2,3}
#   4 dropped -- n_buffer=3 showed no gain over 2 in the earlier sweep, so a
#   deeper pipeline looks unpromising and the axis is not worth 50% more configs.
#
# 3 x 4 x 5 x 2 = 120 configs. OOMs are recorded and the sweep continues.

set -u
ARGS=()
for q in 1 4 16; do
  for kv in 4 8 16 32; do
    for b in 1 2 3 4 8; do
      for n in 2 3; do
        ARGS+=(--variant "num_queries_per_block=${q},num_kv_pages_per_block=${kv},batch_size=${b},n_buffer=${n}")
      done
    done
  done
done

echo "sweeping $(( ${#ARGS[@]} / 2 )) configs"
python run_mla_profile.py \
  --spec decode_f8_kv9216_v3layout \
  --impl v3_native \
  --iterations 10 \
  "${ARGS[@]}"

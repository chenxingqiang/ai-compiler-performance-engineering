#!/bin/bash
# Sequential strict re-validation of the labs fixed this session, run AFTER the
# chapter re-validation finishes (strict's foreign-process check forbids concurrent
# strict runs). Each scope writes its own run-id + updates expectations_4x_gb300.json.
cd /work/ai-performance-engineering/code
# Wait for the in-flight chapter re-validation to release the GPUs.
while pgrep -f gb300_reval_chapters >/dev/null 2>&1; do sleep 15; done
: > /work/logs/reval_rest_progress.log
for spec in \
  "labs/moe_optimization_journey:gb300_reval_moe" \
  "labs/occupancy_tuning:gb300_reval_occ" \
  "labs/train_distributed:gb300_reval_train"; do
  scope="${spec%%:*}"; rid="${spec##*:}"
  source /work/clear_cgroup.sh 2>/dev/null
  echo "$(date -u +%H:%M:%S) START $scope" >> /work/logs/reval_rest_progress.log
  CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 1800 python -m cli.aisp bench run \
    --targets "$scope" --profile none --validity-profile strict \
    --update-expectations --allow-mixed-provenance --run-id "$rid" \
    > "/work/logs/${rid}.log" 2>&1
  echo "$(date -u +%H:%M:%S) END   $scope exit=$?" >> /work/logs/reval_rest_progress.log
done
echo "$(date -u +%H:%M:%S) REVAL_REST_COMPLETE" >> /work/logs/reval_rest_progress.log

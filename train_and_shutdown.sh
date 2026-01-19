#!/bin/bash
# 大规模训练脚本 - 训练完成后自动关机

echo "=========================================="
echo "开始 Attention Consistency Pruning 训练"
echo "=========================================="

cd /projects/domain-adaptation-token-pruning

# 记录开始时间
START_TIME=$(date +%s)
echo "开始时间: $(date)"

# 运行训练（无论成功失败都继续执行）
python main_acp.py 2>&1 | tee outputs/logs/train_$(date +%Y%m%d_%H%M%S).log
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "训练完成!"
else
    echo "训练异常退出 (exit code: $TRAIN_EXIT_CODE)"
fi
echo "结束时间: $(date)"
echo "总耗时: $((DURATION / 3600))小时 $((DURATION % 3600 / 60))分钟"
echo "=========================================="

# 关机（AutoDL 通常是 root 权限）
echo "30秒后关机..."
sleep 30
shutdown -h now

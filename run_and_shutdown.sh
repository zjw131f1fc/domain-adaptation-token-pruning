#!/bin/bash
# AutoDL训练脚本 - 训练完成后自动关机

echo "=========================================="
echo "开始训练: $(date)"
echo "=========================================="

# 运行训练
python main.py

# 获取退出码
EXIT_CODE=$?

echo "=========================================="
echo "训练结束: $(date)"
echo "退出码: $EXIT_CODE"
echo "=========================================="

# 等待5秒，确保日志写入完成
sleep 5

# AutoDL关机命令
if [ $EXIT_CODE -eq 0 ]; then
    echo "训练成功，正在关机..."
else
    echo "训练失败(退出码:$EXIT_CODE)，仍然关机..."
fi

# AutoDL关机方式
shutdown

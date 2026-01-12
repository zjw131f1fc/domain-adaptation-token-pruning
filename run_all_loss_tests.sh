#!/bin/bash
# Loss 收敛性测试脚本
# 顺序测试每个 loss 是否能正常收敛

set +e  # 遇到错误继续执行

cd "$(dirname "$0")"

# 测试列表
TESTS=(
    "task_loss"
    "adv_loss"
    "sparsity_loss"
    "token_count_loss"
    "binarization_loss"
    "disc_loss"
    "all_losses"
)

# 测试步数（可通过参数指定）
NUM_STEPS=${1:-200}

# 结果记录
RESULTS_DIR="outputs/loss_tests"
RESULTS_FILE="$RESULTS_DIR/test_results.txt"
mkdir -p "$RESULTS_DIR"

echo ""
echo "########################################################################"
echo "#                      Loss 收敛性测试                                  #"
echo "########################################################################"
echo ""
echo "测试步数: $NUM_STEPS"
echo "开始时间: $(date)"
echo ""

# 清空结果文件
> "$RESULTS_FILE"

# 记录每个测试的结果
declare -a PASS_TESTS
declare -a FAIL_TESTS

for test_name in "${TESTS[@]}"; do
    echo ""
    echo "========================================"
    echo "  开始测试: $test_name"
    echo "========================================"

    START_TIME=$(date +%s)

    # 运行测试，捕获输出
    OUTPUT=$(python run_loss_test.py "$test_name" "$NUM_STEPS" 2>&1)
    EXIT_CODE=$?

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    # 打印输出
    echo "$OUTPUT"

    # 检查结果（从输出中提取结论）
    if echo "$OUTPUT" | grep -q "✓ 可以收敛\|✓ 均可收敛"; then
        PASS_TESTS+=("$test_name")
        STATUS="✓ PASS"
    else
        FAIL_TESTS+=("$test_name")
        STATUS="✗ FAIL"
    fi

    # 记录到文件
    echo "[$STATUS] $test_name (耗时: ${DURATION}s)" >> "$RESULTS_FILE"

    echo ""
    echo ">>> $test_name: $STATUS (耗时: ${DURATION}s)"

    # 清理显存
    sleep 2
done

# 打印最终汇总
echo ""
echo "########################################################################"
echo "#                         测试结果汇总                                  #"
echo "########################################################################"
echo ""
echo "结束时间: $(date)"
echo ""
echo "通过的测试 (${#PASS_TESTS[@]}/${#TESTS[@]}):"
for t in "${PASS_TESTS[@]}"; do
    echo "  ✓ $t"
done

echo ""
echo "失败的测试 (${#FAIL_TESTS[@]}/${#TESTS[@]}):"
if [ ${#FAIL_TESTS[@]} -eq 0 ]; then
    echo "  (无)"
else
    for t in "${FAIL_TESTS[@]}"; do
        echo "  ✗ $t"
    done
fi

echo ""
echo "------------------------------------------------------------------------"
if [ ${#FAIL_TESTS[@]} -eq 0 ]; then
    echo "  >>> 全部通过! 所有 Loss 均可正常收敛 <<<"
else
    echo "  >>> 存在问题! ${#FAIL_TESTS[@]} 个 Loss 未能收敛 <<<"
fi
echo "------------------------------------------------------------------------"
echo ""
echo "详细结果已保存到: $RESULTS_FILE"

# 关机
echo ""
echo "测试完成，系统将在 10 秒后关机..."
sleep 10
sudo shutdown -h now

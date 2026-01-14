# 训练注意事项

## 1. Task Loss 大小写问题

### 问题描述
VQA数据集的答案通常是小写（如 "blue", "no parking"），但LLaVA等LLM倾向于输出首字母大写（如 "Blue", "No parking"）。

由于tokenizer中 "blue" 和 "Blue" 是不同的token，即使语义正确，cross entropy loss也会很大（10+），导致：
- Task loss信号不准确
- 梯度被大小写差异主导，而不是语义正确性
- 模型被迫学习"输出小写"而不是"输出正确答案"

### 解决方案
在 `method/utils.py` 的 `compute_task_loss_batch` 函数中，将GT答案转为首字母大写：

```python
answer = answers[i].capitalize()  # "blue" -> "Blue"
```

### 验证方法
打印GT和预测对比：
```
Blue Blue loss=0.3xxx  # 正确：大小写匹配，loss小
blue Blue loss=10.xxx  # 错误：大小写不匹配，loss虚高
```

### 注意
- evaluate时通常会做大小写归一化比较，所以准确率不受影响
- 但training时的cross entropy是token级别精确匹配，必须处理大小写

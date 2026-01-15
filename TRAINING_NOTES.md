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

## 2. Batch Padding方向问题

### 问题描述
LLaMA/LLaVA默认使用左padding（在序列开头加PAD），但训练时的位置计算（vision_pos, question_pos, answer_pos）假设是右padding。

当使用左padding时，较短样本的answer位置会指向错误的位置，导致：
- loss异常高（10+、15+）
- 预测结果是随机字符（如 `.`, `)`, `</s>`）

### 解决方案
在 `engine/backbones/impl/mllm/llava.py` 加载processor后，设置右padding：

```python
self.processor.tokenizer.padding_side = "right"
```

### 验证方法
检查 `seq_len` 和 `sample_len` 的关系：
- 左padding（错误）：被padding的样本预测结果异常
- 右padding（正确）：所有样本预测结果正常

### 注意
- 这个设置影响所有batch处理，包括vision_pos、question_pos、answer_pos
- 推理时如果需要左padding（用于生成），需要单独处理

"""测试question embeddings提取逻辑是否正确"""
import torch

def test_question_extraction():
    print("=" * 80)
    print("测试 Question Embeddings 提取逻辑")
    print("=" * 80)

    # 模拟embeddings结构
    # "USER: " (3 tokens) + vision (576) + "\nWhat color is the car?\nASSISTANT: " (10 tokens) + "red" (1 token)
    seq_len = 3 + 576 + 10 + 1  # = 590
    d_model = 4096

    embeddings = torch.randn(1, seq_len, d_model)

    # 模拟位置
    vision_start = 3
    vision_end = 3 + 576 - 1  # = 578
    answer_start = -1  # 负索引，相对于末尾

    print(f"\n[模拟数据]")
    print(f"总长度: {seq_len}")
    print(f"Vision位置: [{vision_start}:{vision_end+1}] (共{vision_end - vision_start + 1}个tokens)")
    print(f"Answer位置: {answer_start} (负索引)")

    # 转换answer负索引为正索引
    if answer_start < 0:
        answer_start_abs = seq_len + answer_start
    else:
        answer_start_abs = answer_start

    print(f"Answer位置(转正): {answer_start_abs}")

    # 提取question (原始逻辑 - 错误1)
    print(f"\n[错误的提取方式1] (只取vision之前)")
    question_wrong1 = embeddings[:, 0:vision_start, :]
    print(f"Question shape: {question_wrong1.shape}")
    print(f"只包含: 'USER: ' (3个tokens) ❌")

    # 提取question (原始逻辑 - 错误2)
    print(f"\n[错误的提取方式2] (拼接USER和question)")
    question_part1 = embeddings[:, 0:vision_start, :]  # "USER: "
    question_part2 = embeddings[:, vision_end+1:answer_start_abs, :]  # "\nWhat color...\nASSISTANT: "
    question_wrong2 = torch.cat([question_part1, question_part2], dim=1)
    print(f"Question shape: {question_wrong2.shape}")
    print(f"包含: 'USER: ' + question + 'ASSISTANT: ' (13个tokens) ❌")

    # 提取question (修复后 - 正确)
    print(f"\n[正确的提取方式] (只取question本身，不包含USER:)")
    question_correct = embeddings[:, vision_end+1:answer_start_abs, :]  # "\nWhat color...\nASSISTANT: "
    print(f"Question shape: {question_correct.shape}")
    print(f"只包含: '\\nWhat color is the car?\\nASSISTANT: ' (10个tokens) ✓")

    # 验证
    expected_len = 10  # 只有 question + "ASSISTANT: "
    assert question_correct.shape[1] == expected_len, f"长度不匹配: {question_correct.shape[1]} != {expected_len}"
    print(f"\n✓ 验证通过! Question只包含{expected_len}个tokens (不含'USER: ')")

    # 测试evaluation场景 (answer=None)
    print("\n" + "=" * 80)
    print("测试 Evaluation 场景 (answer=None)")
    print("=" * 80)

    seq_len_eval = 3 + 576 + 10  # 没有answer
    embeddings_eval = torch.randn(1, seq_len_eval, d_model)

    print(f"\n[模拟数据]")
    print(f"总长度: {seq_len_eval}")
    print(f"Vision位置: [{vision_start}:{vision_end+1}]")
    print(f"Answer: None (evaluation模式)")

    # 提取question (evaluation - 错误)
    print(f"\n[错误的提取方式] (拼接两部分)")
    question_part1 = embeddings_eval[:, 0:vision_start, :]  # "USER: "
    question_part2 = embeddings_eval[:, vision_end+1:, :]  # "\nWhat color...\nASSISTANT:"
    question_eval_wrong = torch.cat([question_part1, question_part2], dim=1)
    print(f"Question shape: {question_eval_wrong.shape}")
    print(f"包含: 'USER: ' + question + 'ASSISTANT: ' (13个tokens) ❌")

    # 提取question (evaluation - 正确)
    print(f"\n[正确的提取方式] (只取question本身)")
    question_eval = embeddings_eval[:, vision_end+1:, :]  # "\nWhat color...\nASSISTANT:"
    print(f"Question shape: {question_eval.shape}")
    print(f"只包含: '\\nWhat color is the car?\\nASSISTANT: ' (10个tokens) ✓")

    expected_len_eval = 10
    assert question_eval.shape[1] == expected_len_eval
    print(f"\n✓ 验证通过! Question只包含{expected_len_eval}个tokens (不含'USER: ')")

    print("\n" + "=" * 80)
    print("✓ 所有测试通过!")
    print("=" * 80)

if __name__ == "__main__":
    test_question_extraction()

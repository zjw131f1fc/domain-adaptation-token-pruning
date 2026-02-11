#!/usr/bin/env python
"""分析数据集中答案的分布情况

用法:
    python scripts/analyze_dataset_answers.py
"""

import os
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path
from collections import Counter

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_dataset_samples(max_samples=None):
    """加载数据集"""
    from engine.configs.loader import load_config
    from engine.datas.loader import load_dataset

    config = load_config(override_file="configs/vision_token_pruning.yaml")

    if max_samples:
        config.dataset_settings['split'] = {'train': max_samples, 'test': 100}

    data_bundle = load_dataset(config)
    train_dataset = data_bundle['splits']['train']

    return train_dataset


def analyze_answers(dataset):
    """分析答案分布"""
    print(f"\n{'='*60}")
    print(f"数据集大小: {len(dataset)}")
    print(f"{'='*60}\n")

    # 统计答案
    answer_counter = Counter()
    answer_lengths = []
    special_answers = []  # 特殊答案（None, N/A 等）

    for i, sample in enumerate(dataset):
        answer = sample.get('answer', '')
        answer_lower = answer.lower().strip()

        answer_counter[answer_lower] += 1
        answer_lengths.append(len(answer))

        # 检查特殊答案
        if answer_lower in ['none', 'n/a', 'na', 'null', '', 'unknown', 'unanswerable']:
            special_answers.append({
                'index': i,
                'question': sample.get('question', ''),
                'answer': answer
            })

    # 打印最常见的答案
    print("【最常见的 30 个答案】")
    print("-" * 40)
    for answer, count in answer_counter.most_common(30):
        pct = count / len(dataset) * 100
        print(f"  '{answer}': {count} ({pct:.2f}%)")

    # 打印答案长度统计
    print(f"\n【答案长度统计】")
    print("-" * 40)
    print(f"  最短: {min(answer_lengths)}")
    print(f"  最长: {max(answer_lengths)}")
    print(f"  平均: {sum(answer_lengths)/len(answer_lengths):.2f}")

    # 长度分布
    length_counter = Counter(answer_lengths)
    print(f"\n  长度分布 (前10):")
    for length, count in sorted(length_counter.items())[:10]:
        pct = count / len(dataset) * 100
        print(f"    长度 {length}: {count} ({pct:.2f}%)")

    # 打印特殊答案
    print(f"\n【特殊答案 (None/N/A 等)】")
    print("-" * 40)
    print(f"  总数: {len(special_answers)}")
    if special_answers:
        print(f"\n  示例 (前10个):")
        for item in special_answers[:10]:
            print(f"    [{item['index']}] Q: {item['question'][:50]}...")
            print(f"         A: '{item['answer']}'")

    # 统计单词数
    print(f"\n【答案单词数统计】")
    print("-" * 40)
    word_counts = [len(sample.get('answer', '').split()) for sample in dataset]
    word_counter = Counter(word_counts)
    for n_words, count in sorted(word_counter.items())[:10]:
        pct = count / len(dataset) * 100
        print(f"  {n_words} 个单词: {count} ({pct:.2f}%)")

    return answer_counter, special_answers


def analyze_tokenization(dataset, max_samples=100):
    """分析 tokenize 后的答案"""
    from transformers import AutoProcessor

    print(f"\n{'='*60}")
    print("【Tokenization 分析】")
    print(f"{'='*60}\n")

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    tokenizer = processor.tokenizer

    token_lengths = []
    mismatch_samples = []

    for i, sample in enumerate(dataset[:max_samples]):
        answer = sample.get('answer', '').capitalize()

        # 直接 tokenize
        answer_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
        token_lengths.append(len(answer_ids))

        # 检查带空格和不带空格的区别
        answer_with_space = " " + answer
        answer_ids_with_space = tokenizer(answer_with_space, add_special_tokens=False)['input_ids']

        # 构建完整 prompt 并 tokenize
        prompt = f"USER: <image>\nTest question\nASSISTANT: {answer}"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']

        # 找到 ASSISTANT: 后面的 token
        assistant_str = "\nASSISTANT:"
        assistant_ids = tokenizer(assistant_str, add_special_tokens=False)['input_ids']
        if assistant_ids[0] == 29871:
            assistant_ids = assistant_ids[1:]

        # 在 prompt_ids 中找 assistant_ids
        found_pos = None
        for j in range(len(prompt_ids) - len(assistant_ids) + 1):
            if prompt_ids[j:j+len(assistant_ids)] == assistant_ids:
                found_pos = j + len(assistant_ids)
                break

        if found_pos is not None:
            actual_answer_ids = prompt_ids[found_pos:]

            # 检查是否匹配
            if answer_ids != actual_answer_ids:
                mismatch_samples.append({
                    'index': i,
                    'answer': answer,
                    'direct_ids': answer_ids,
                    'actual_ids': actual_answer_ids,
                    'direct_decoded': tokenizer.decode(answer_ids),
                    'actual_decoded': tokenizer.decode(actual_answer_ids),
                })

    # 打印 token 长度统计
    print("Token 长度分布:")
    length_counter = Counter(token_lengths)
    for length, count in sorted(length_counter.items())[:10]:
        pct = count / max_samples * 100
        print(f"  {length} tokens: {count} ({pct:.2f}%)")

    # 打印不匹配的样本
    print(f"\n不匹配样本数: {len(mismatch_samples)} / {max_samples}")
    if mismatch_samples:
        print("\n不匹配示例 (前5个):")
        for item in mismatch_samples[:5]:
            print(f"  [{item['index']}] answer='{item['answer']}'")
            print(f"       direct_ids={item['direct_ids']} -> '{item['direct_decoded']}'")
            print(f"       actual_ids={item['actual_ids']} -> '{item['actual_decoded']}'")


def main():
    print("加载数据集...")
    dataset = load_dataset_samples(max_samples=10000)

    # 分析答案分布
    answer_counter, special_answers = analyze_answers(dataset)

    # 分析 tokenization
    analyze_tokenization(dataset, max_samples=500)

    print(f"\n{'='*60}")
    print("分析完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""MMB 和 SQA 数据集长度分布对比"""

import os
import sys
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from transformers import AutoProcessor
from collections import Counter

def check_length_distribution():
    print("=" * 70)
    print("MMB 和 SQA 数据集长度分布对比")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        cache_dir="/data/users/zjw/huggingface_cache"
    )

    results = {}

    # ========== MMB ==========
    print("\n[1/2] 处理 MMB 数据集...")
    mmb_lengths = []
    for sub in ['cn', 'en']:
        ds = load_dataset("lmms-lab/MMBench", sub, split="dev")
        for i in range(len(ds)):
            item = ds[i]
            q_raw = item['question']
            if not q_raw or not str(q_raw).strip():
                continue
            if not item['answer'] or not str(item['answer']).strip():
                continue
            opt_lines = [f"A. {item['A']}", f"B. {item['B']}", f"C. {item['C']}", f"D. {item['D']}"]
            instr = "Choose the correct answer from A/B/C/D and output only one letter (A, B, C, or D)."
            full_q = f"{q_raw}\n" + "\n".join(opt_lines) + f"\n{instr}"
            eos = processor.tokenizer.eos_token or "</s>"
            prompt = f"USER: <image>\n{full_q}\nASSISTANT: {item['answer']}{eos}"
            tokens = processor.tokenizer(prompt, return_tensors="pt")
            seq_len = tokens['input_ids'].shape[1] + 576
            mmb_lengths.append(seq_len)
    results['MMB'] = mmb_lengths
    print(f"  样本数: {len(mmb_lengths)}")

    # ========== SQA ==========
    print("\n[2/2] 处理 SQA 数据集...")
    sqa_lengths = []
    ds = load_dataset("derek-thomas/ScienceQA", split="train")
    for i in range(len(ds)):
        item = ds[i]
        if item['image'] is None:
            continue
        question = item['question']
        choices = item['choices']
        opt_lines = [f"({chr(ord('A') + idx)}) {opt}" for idx, opt in enumerate(choices)]
        instr = "Answer with the option letter (A, B, C, or D) at the end."
        hint = item.get('hint', '') or ''
        lecture = item.get('lecture', '') or ''
        prompt_parts = []
        if lecture.strip():
            prompt_parts.append(f"Background: {lecture.strip()}")
        if hint.strip():
            prompt_parts.append(f"Hint: {hint.strip()}")
        prompt_parts.append(question)
        prompt_parts.append("\n".join(opt_lines))
        prompt_parts.append(instr)
        full_q = "\n".join(prompt_parts)
        answer = chr(ord('A') + item['answer'])
        eos = processor.tokenizer.eos_token or "</s>"
        prompt = f"USER: <image>\n{full_q}\nASSISTANT: {answer}{eos}"
        tokens = processor.tokenizer(prompt, return_tensors="pt")
        seq_len = tokens['input_ids'].shape[1] + 576
        sqa_lengths.append(seq_len)
    results['SQA'] = sqa_lengths
    print(f"  样本数: {len(sqa_lengths)}")

    # ========== 统计输出 ==========
    print("\n" + "=" * 70)
    print("长度分布统计")
    print("=" * 70)

    bins = [(0, 700), (700, 800), (800, 900), (900, 1000), (1000, 1100),
            (1100, 1200), (1200, 1300), (1300, 1400), (1400, 1500), (1500, 2000)]

    for name, lengths in results.items():
        print(f"\n【{name}】")
        print(f"  样本数: {len(lengths)}")
        print(f"  最小: {min(lengths)}, 最大: {max(lengths)}, 平均: {sum(lengths)/len(lengths):.1f}")
        print(f"\n  长度分布:")
        for lo, hi in bins:
            count = sum(1 for x in lengths if lo <= x < hi)
            pct = count / len(lengths) * 100
            bar = '█' * int(pct / 2)
            print(f"    {lo:4d}-{hi:4d}: {count:5d} ({pct:5.1f}%) {bar}")

        # 超过阈值统计
        print(f"\n  超过阈值:")
        for threshold in [1024, 1200, 1400, 1536]:
            count = sum(1 for x in lengths if x > threshold)
            pct = count / len(lengths) * 100
            print(f"    >{threshold}: {count:5d} ({pct:5.1f}%)")

if __name__ == "__main__":
    check_length_distribution()

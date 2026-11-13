"""LoRA 微调 Qwen2.5-0.5B-Instruct：让小型本地模型"记住"课程知识库。

用法:  python finetune.py            # CPU 可跑，约 10-30 分钟
输出:  models/qwen-lora/  (LoRA adapter，配合基座模型做推理)
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "finetune_data.jsonl"
ADAPTER_DIR = BASE_DIR / "models" / "qwen-lora"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LEN = 1280  # 蒸馏样本含 RAG 上下文（system 提示较长）
PROMPT_TEMPLATE = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


class QADataset(Dataset):
    def __init__(self, path: Path, tokenizer):
        self.rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            if ex.get("system"):
                # 与本地推理完全一致的提示词（apply_chat_template）
                prompt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": ex["system"]},
                     {"role": "user", "content": ex["prompt"]}],
                    tokenize=False, add_generation_prompt=True)
            else:
                prompt = PROMPT_TEMPLATE.format(q=ex["prompt"])
            full = prompt + ex["completion"] + "<|im_end|>"
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:MAX_LEN]
            labels = [-100] * len(p_ids) + f_ids[len(p_ids):]  # 只对回答部分计算损失
            self.rows.append({"input_ids": f_ids, "labels": labels})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def main():
    if not DATA_FILE.exists():
        raise SystemExit("未找到数据集，请先运行: python build_dataset.py")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model.config.use_cache = False

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = QADataset(DATA_FILE, tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    args = TrainingArguments(
        output_dir=str(BASE_DIR / "models" / "train_out"),
        per_device_train_batch_size=2,   # 长序列样本（含上下文），减小 batch 防爆内存
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=lambda b: collate(b, pad_id))
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"\n训练完成，LoRA adapter 已保存到 {ADAPTER_DIR}")


if __name__ == "__main__":
    main()

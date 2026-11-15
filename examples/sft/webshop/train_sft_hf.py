# Copyright 2026 BiPACE contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-turn SFT via HuggingFace Trainer.

Replacement for verl-agent's fsdp_sft_trainer that has a known TODO around
checkpoint management. HF Trainer's `save_steps` is reliable.

Trains only on assistant tokens (other turns masked to -100), matching the
behaviour of verl's MultiTurnSFTDataset.
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class MultiTurnSFTDataset(Dataset):
    """Per-turn loss masking: only assistant spans contribute to loss."""

    def __init__(self, parquet_path: str, tokenizer, max_length: int = 8192, truncation: str = "right"):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert truncation in ("right", "left")
        self.truncation = truncation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        messages = [dict(m) for m in self.df.iloc[idx]["messages"]]
        tok = self.tokenizer

        full_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        labels = [-100] * len(full_ids)

        prev_len = 0
        for i, msg in enumerate(messages):
            prefix_ids = tok.apply_chat_template(messages[: i + 1], tokenize=True, add_generation_prompt=False)
            end_len = len(prefix_ids)
            if msg["role"] == "assistant":
                for k in range(prev_len, min(end_len, len(full_ids))):
                    labels[k] = full_ids[k]
            prev_len = end_len

        # Right-truncate (drop tail) by default — losing the last assistant
        # span is bad, so for max_length-overrun we prefer left-truncate.
        if len(full_ids) > self.max_length:
            if self.truncation == "right":
                full_ids = full_ids[: self.max_length]
                labels = labels[: self.max_length]
            else:
                full_ids = full_ids[-self.max_length :]
                labels = labels[-self.max_length :]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }


@dataclass
class PadCollator:
    """Right-pad input_ids / attention_mask / labels to the longest seq in the
    batch. labels pad with -100 so they don't contribute to loss.
    """

    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(b["input_ids"]) for b in batch)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids, attn, labels = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = max_len - n
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--per_device_bs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--truncation", default="right", choices=["right", "left"])
    p.add_argument("--save_steps", type=int, default=40)
    p.add_argument("--save_total_limit", type=int, default=None)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", default="eto-sft-webshop")
    p.add_argument("--run_name", default="qwen2.5-1.5b-hf-2gpu")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False  # required for grad checkpointing

    train_ds = MultiTurnSFTDataset(args.train_file, tok, args.max_length, args.truncation)
    eval_ds = MultiTurnSFTDataset(args.eval_file, tok, args.max_length, args.truncation)

    collator = PadCollator(tokenizer=tok)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch",
        load_best_model_at_end=False,
        report_to=["wandb"],
        run_name=args.run_name,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")
    tok.save_pretrained(f"{args.output_dir}/final")


if __name__ == "__main__":
    main()

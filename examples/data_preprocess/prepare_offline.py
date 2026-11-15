# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""
Offline / network-isolated equivalent of examples/data_preprocess/prepare.py.

The upstream prepare.py downloads `hiyouga/geometry3k` from HF Hub solely to
get a row count of the right shape — its own docstring says "We do NOT use
the data in 'hiyouga/geometry3k', instead we only use it to indicate the
modality and the data size." This script synthesizes the same manifest
directly without any network calls.
"""

import argparse
import os

import pandas as pd


def make_row(mode: str, split: str, idx: int) -> dict:
    prompt_content = "<image>" if mode == "visual" else ""
    row = {
        "data_source": mode,
        "prompt": [{"role": "user", "content": prompt_content}],
        "ability": "agent",
        "extra_info": {"split": split, "index": idx},
    }
    if mode == "visual":
        row["images"] = []  # placeholder; agent rollout supplies real images
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="text", choices=["visual", "text"])
    p.add_argument("--local_dir", default="~/data/verl-agent/")
    p.add_argument("--train_data_size", default=256, type=int)
    p.add_argument("--val_data_size", default=256, type=int)
    args = p.parse_args()

    out_dir = os.path.expanduser(os.path.join(args.local_dir, args.mode))
    os.makedirs(out_dir, exist_ok=True)

    train_rows = [make_row(args.mode, "train", i) for i in range(args.train_data_size)]
    test_rows = [make_row(args.mode, "test", i) for i in range(args.val_data_size)]

    pd.DataFrame(train_rows).to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    pd.DataFrame(test_rows).to_parquet(os.path.join(out_dir, "test.parquet"), index=False)

    print(f"[prepare_offline] mode={args.mode}")
    print(f"[prepare_offline] wrote {len(train_rows)} train rows -> {out_dir}/train.parquet")
    print(f"[prepare_offline] wrote {len(test_rows)}  test  rows -> {out_dir}/test.parquet")


if __name__ == "__main__":
    main()

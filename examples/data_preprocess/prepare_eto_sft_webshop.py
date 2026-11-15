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
"""Download the ETO webshop expert-trajectory SFT split and convert it to
the multi-turn parquet format verl's MultiTurnSFTDataset expects.

Source: https://huggingface.co/datasets/agent-eto/eto-sft-trajectory
Split:  "webshop" (BFS expert trajectories on the WebShop benchmark)

The source uses ShareGPT format ({"from": "human"|"gpt", "value": str}).
We rename to OpenAI chat format ({"role": "user"|"assistant", "content": str})
and store the column as `messages` so it works out of the box with
`data.multiturn.enable=true data.multiturn.messages_key=messages`.

Usage:

    python -m examples.data_preprocess.prepare_eto_sft_webshop \
        --local_dir ~/data/verl-agent/eto_sft_webshop/ \
        --val_ratio 0.05
"""

import argparse
import os
import random

from datasets import load_dataset

_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def _convert_sharegpt_to_messages(conversations):
    msgs = []
    for turn in conversations:
        role = _ROLE_MAP.get(turn["from"], turn["from"])
        msgs.append({"role": role, "content": turn["value"]})
    return msgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/verl-agent/eto_sft_webshop/")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Fraction held out for eval. 5% by default — the dataset has only ~1.9k examples so we keep most for training.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.local_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("[prepare] downloading agent-eto/eto-sft-trajectory webshop_sft.json…")
    # Load the webshop file directly. Loading the whole repo as a dataset
    # fails because alfworld/sciworld splits have extra columns
    # ({reward, source}) that don't unify cleanly.
    ds = load_dataset(
        "json",
        data_files="hf://datasets/agent-eto/eto-sft-trajectory/data/webshop_sft.json",
        split="train",
    )
    print(f"[prepare] loaded {len(ds)} trajectories")

    ds = ds.map(
        lambda ex: {"messages": _convert_sharegpt_to_messages(ex["conversations"])},
        remove_columns=[c for c in ds.column_names if c != "id"],
    )

    # Deterministic 95/5 split
    n = len(ds)
    rng = random.Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = max(1, int(n * args.val_ratio))
    val_idx = set(indices[:n_val])
    train_idx = [i for i in indices if i not in val_idx]

    train_ds = ds.select(train_idx)
    val_ds = ds.select(sorted(val_idx))

    train_path = os.path.join(out_dir, "train.parquet")
    val_path = os.path.join(out_dir, "test.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    sample_msgs = train_ds[0]["messages"]
    print(f"[prepare] wrote {train_path} ({len(train_ds)} rows)")
    print(f"[prepare] wrote {val_path}  ({len(val_ds)} rows)")
    print(f"[prepare] sample: {len(sample_msgs)} turns; roles = {[m['role'] for m in sample_msgs[:6]]}…")


if __name__ == "__main__":
    main()

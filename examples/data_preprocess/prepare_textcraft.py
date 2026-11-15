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
"""Prepare the TextCraft "dataset" for verl-agent.

TextCraft is a pure-Python crafting simulator (vendored MIT-licensed source under
``agent_system/environments/env_package/textcraft/textcraft_src/``); there is no
pre-existing parquet to download. This script:

  1. Reads the bundled depth-annotated goal catalog
     (``textcraft_goal_index.json``, 397 goals at min_depth ≥ 2) and the
     committed 80/20 depth-stratified split (``textcraft_splits.json``).
  2. Emits ``train.parquet`` / ``test.parquet`` placeholders with one row per
     requested sample. verl-agent only uses the parquet to signal modality +
     row count; actual env state comes from the Ray-vectorised TextCraft
     workers at rollout time.

Usage on the dev box:

    python -m examples.data_preprocess.prepare_textcraft \
        --train_data_size 16 --val_data_size 64

Then upload the output dir to the HPC:

    scp -r ~/data/verl-agent/textcraft/ user@hpc:~/data/verl-agent/
"""

import argparse
import json
import os

import pandas as pd


def _env_pkg_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "agent_system", "environments", "env_package", "textcraft"))


def _load_split():
    path = os.path.join(_env_pkg_root(), "textcraft_splits.json")
    with open(path) as f:
        return json.load(f)


def _load_goal_index():
    path = os.path.join(_env_pkg_root(), "textcraft_goal_index.json")
    with open(path) as f:
        rows = json.load(f)
    return {r["goal_item"]: int(r["depth"]) for r in rows}


def _placeholder_rows(split_name: str, n: int):
    # Schema only needs to satisfy the downstream verl-agent loader, which
    # inspects the column names + row count.
    return [
        {
            "data_source": "textcraft",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": split_name, "index": idx},
        }
        for idx in range(n)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/verl-agent/textcraft/")
    parser.add_argument("--train_data_size", type=int, default=16, help="Number of placeholder training rows; must be ≤ env.train batch * iterations.")
    parser.add_argument("--val_data_size", type=int, default=64)
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.local_dir)
    os.makedirs(out_dir, exist_ok=True)

    split = _load_split()
    goal_index = _load_goal_index()

    # Sanity: every goal in the split must have a depth annotation.
    missing = [g for g in split["train"] + split["val"] if g not in goal_index]
    if missing:
        raise RuntimeError(f"{len(missing)} goals in the split are missing depth annotations (first few: {missing[:5]}). Regenerate textcraft_goal_index.json.")

    # Echo the committed split + depth distribution.
    from collections import Counter

    train_depths = Counter(goal_index[g] for g in split["train"])
    val_depths = Counter(goal_index[g] for g in split["val"])
    summary = {
        "n_train_goals": len(split["train"]),
        "n_val_goals": len(split["val"]),
        "train_depth_dist": dict(sorted(train_depths.items())),
        "val_depth_dist": dict(sorted(val_depths.items())),
        "split_strategy": split.get("split_strategy", "unknown"),
    }
    print(f"[prepare] split summary: {summary}")
    summary_path = os.path.join(out_dir, "splits.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[prepare] wrote {summary_path}")

    train_df = pd.DataFrame(_placeholder_rows("train", args.train_data_size))
    test_df = pd.DataFrame(_placeholder_rows("test", args.val_data_size))
    train_path = os.path.join(out_dir, "train.parquet")
    test_path = os.path.join(out_dir, "test.parquet")
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    print(f"[prepare] wrote {train_path} ({len(train_df)} rows)")
    print(f"[prepare] wrote {test_path}  ({len(test_df)} rows)")
    print(f"[prepare] Done. Upload this dir to HPC:\n    {out_dir}")


if __name__ == "__main__":
    main()

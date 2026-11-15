#!/usr/bin/env python3
"""Build AppWorld SFT cold-start parquet from expert ground-truth solutions.

Each train task contributes one sample:
  - prompt   = APPWORLD_TEMPLATE_NO_HIS rendered with the task's supervisor info
               and instruction (same template the RL stage uses for reset);
  - response = a <think> placeholder + the expert's `solution()` body wrapped in
               <code>...</code> (the same format appworld_projection extracts).

Output: /home/work/data/verl-agent/appworld_sft/{train,val}.parquet
        (columns prompt / response, fsdp_sft_trainer compatible).
"""
from __future__ import annotations

import ast
import json
import sys
import textwrap
from pathlib import Path

import pandas as pd

REPO = Path("/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE")
sys.path.insert(0, str(REPO))
from agent_system.environments.prompts.appworld import APPWORLD_TEMPLATE_NO_HIS  # noqa: E402

DATA_ROOT = Path("/mnt/tidalfs-hssh01/data")
TASKS = DATA_ROOT / "tasks"
DATASETS = DATA_ROOT / "datasets"
OUT_DIR = Path("/home/work/data/verl-agent/appworld_sft")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THINK = "I will interact with the available apps via their APIs to complete the task."


def extract_solution_body(source: str) -> str | None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solution":
            lines = source.splitlines()
            start = node.body[0].lineno - 1  # 0-indexed first body line
            end = node.end_lineno  # inclusive 1-indexed last line
            body = "\n".join(lines[start:end])
            return textwrap.dedent(body).strip()
    return None


def build_rows() -> list[dict]:
    task_ids = [ln.strip() for ln in (DATASETS / "train.txt").read_text().splitlines() if ln.strip()]
    rows = []
    skipped = []
    for tid in task_ids:
        task_dir = TASKS / tid
        specs = json.loads((task_dir / "specs.json").read_text())
        sol = (task_dir / "ground_truth" / "compiled_solution.py").read_text()
        body = extract_solution_body(sol)
        if body is None:
            skipped.append(tid)
            continue
        supervisor = specs["supervisor"]
        prompt = APPWORLD_TEMPLATE_NO_HIS.format(
            supervisor_first_name=supervisor["first_name"],
            supervisor_last_name=supervisor["last_name"],
            supervisor_email=supervisor["email"],
            supervisor_phone_number=supervisor["phone_number"],
            task_description=specs["instruction"],
        ).strip()
        response = f"<think>\n{THINK}\n</think>\n<code>\n{body}\n</code>"
        rows.append({"prompt": prompt, "response": response})
    return rows, skipped


def main() -> None:
    rows, skipped = build_rows()
    df = pd.DataFrame(rows)
    n = len(df)
    n_val = max(1, n // 10)
    val = df.sample(n_val, random_state=0)
    train = df.drop(val.index)
    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    val.to_parquet(OUT_DIR / "val.parquet", index=False)
    print(f"[prep_appworld_sft] train={len(train)} val={len(val)} skipped={len(skipped)}")
    if skipped:
        print(f"[prep_appworld_sft] skipped tasks: {skipped}")


if __name__ == "__main__":
    main()

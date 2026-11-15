#!/usr/bin/env python3
"""Build AppWorld MULTI-STEP SFT parquet by replaying expert solutions.

Run in the pydantic-v1 venv (needs the full appworld client).  For each train
task it:
  1. starts a remote AppWorld client against a dedicated server;
  2. extracts the expert `solution()` body and splits it into top-level
     statements, grouped into small chunks (token-budgeted);
  3. executes chunk-by-chunk in the SAME session, capturing each result;
  4. emits one SFT sample per chunk: prompt = APPWORLD template + history of
     (code_i, result_i), response = next <code> chunk.

Output: /home/work/data/verl-agent/appworld_sft_multistep/{train,val}.parquet
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pandas as pd

REPO = Path("/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE")

# Load only the prompt-string module (avoid importing agent_system's heavy
# __init__ which needs omegaconf/ray, unavailable in this pydantic-v1 venv).
_prompts_path = REPO / "agent_system/environments/prompts/appworld.py"
_spec = importlib.util.spec_from_file_location("_appworld_prompts", _prompts_path)
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)
APPWORLD_TEMPLATE = _prompts.APPWORLD_TEMPLATE

DATA_ROOT = Path("/mnt/tidalfs-hssh01/data")
TASKS = DATA_ROOT / "tasks"
DATASETS = DATA_ROOT / "datasets"
OUT_DIR = Path("/home/work/data/verl-agent/appworld_sft_multistep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SERVER = "http://0.0.0.0:8162"
MAX_CHUNK_CHARS = 700  # keep each chunk comfortably under the response budget


def top_level_statements(source: str) -> list[str]:
    tree = ast.parse(source)
    stmts = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solution":
            for s in node.body:
                stmts.append(ast.unparse(s))
    return stmts


def chunk_statements(statements: list[str]) -> list[str]:
    chunks, cur = [], []
    cur_len = 0
    for s in statements:
        if cur and cur_len + len(s) > MAX_CHUNK_CHARS:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += len(s)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def replay_task(task_id: str) -> list[dict] | None:
    from appworld import AppWorld, update_root

    update_root("/mnt/tidalfs-hssh01")

    specs = json.loads((TASKS / task_id / "specs.json").read_text())
    sol_src = (TASKS / task_id / "ground_truth" / "compiled_solution.py").read_text()
    stmts = top_level_statements(sol_src)
    if not stmts:
        return None
    chunks = chunk_statements(stmts)

    w = AppWorld(task_id=task_id, experiment_name="sft_replay", remote_environment_url=SERVER)
    history: list[str] = []
    samples = []
    for code in chunks:
        prompt = APPWORLD_TEMPLATE.format(
            supervisor_first_name=specs["supervisor"]["first_name"],
            supervisor_last_name=specs["supervisor"]["last_name"],
            supervisor_email=specs["supervisor"]["email"],
            supervisor_phone_number=specs["supervisor"]["phone_number"],
            task_description=specs["instruction"],
            step_count=len(history) // 2,
            history_length=len(history) // 2,
            action_history="".join(history),
            current_step=len(history) // 2 + 1,
        ).strip()
        response = f"<think>\nInteracting with apps via their APIs.\n</think>\n<code>\n{code}\n</code>"
        samples.append({"prompt": prompt, "response": response})
        try:
            result = w.execute(code)
        except Exception as e:  # pragma: no cover
            result = f"Execution failed: {e}"
        history.append(f"\nCode {len(history)//2 + 1}:\n{code}\n\nResult {len(history)//2 + 1}:\n{result}\n")
    w.close()
    return samples


def main() -> None:
    task_ids = [ln.strip() for ln in (DATASETS / "train.txt").read_text().splitlines() if ln.strip()]
    rows = []
    skipped = []
    for i, tid in enumerate(task_ids, 1):
        try:
            samples = replay_task(tid)
            if samples:
                rows.extend(samples)
            else:
                skipped.append(tid)
        except Exception as e:
            skipped.append(f"{tid}:{type(e).__name__}")
        if i % 5 == 0:
            print(f"[prep] {i}/{len(task_ids)} tasks, {len(rows)} samples", flush=True)
    df = pd.DataFrame(rows)
    n_val = max(1, len(df) // 10)
    val = df.sample(n_val, random_state=0)
    train = df.drop(val.index)
    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    val.to_parquet(OUT_DIR / "val.parquet", index=False)
    print(f"[prep] train={len(train)} val={len(val)} skipped={len(skipped)}")
    if skipped:
        print(f"[prep] skipped: {skipped}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract ALFWorld per-task / WebShop score+success metrics from training logs.

Protocol (matches paper_cvpr2027/tables.tex):
  * peak val/success_rate within the stated budget (200 steps for 1.5B, 150 for 7B);
  * the six ALFWorld per-task cells are read at the SAME step as the overall peak
    (not independently maximized);
  * WebShop Score and Success are independently maximized within budget.

Outputs a JSON and a printed table. Run: python3 scripts/extract_subtask_metrics.py
"""
import glob
import json
import os
import re
from collections import defaultdict

LOG_DIR = "/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log"
OUT = "/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/paper_cvpr2027/subtask_metrics.json"

ALF_TASKS = [
    ("Pick", "pick_and_place_success_rate"),
    ("Look", "look_at_obj_in_light_success_rate"),
    ("Clean", "pick_clean_then_place_in_recep_success_rate"),
    ("Heat", "pick_heat_then_place_in_recep_success_rate"),
    ("Cool", "pick_cool_then_place_in_recep_success_rate"),
    ("Pick2", "pick_two_obj_and_place_success_rate"),
]

# (key, scale, budget, [filename glob patterns], seed)
ALF_MANIFEST = [
    ("GRPO", "1.5B", 200, "ladder_grpo_1.5b_alfworld_seed{seed}_thisnode_*.log"),
    ("GiGPO", "1.5B", 200, "ladder_gigpo_1.5b_alfworld_seed{seed}_thisnode_*.log"),
    ("GiGPO", "7B", 150, "ladder_gigpo_7b_alfworld_seed{seed}_thisnode_*.log"),
    ("BiPACE-Q", "1.5B", 200, "bipace_1.5b_alfworld_official_bipaceq_seed{seed}_thisnode_*.log"),
    ("BiPACE-Q", "7B", 150, "bipace_7b_alfworld_official_bipaceq_rawanchor_seed{seed}_thisnode_*.log"),
    # PABH 1.5B seed0 historically logged without a seed suffix.
    ("PABH", "1.5B", 200, "bipace_1.5b_alfworld_bisimhead_g_prmanchor_seed{seed}_thisnode_*.log",
     "bipace_1.5b_alfworld_bisimhead_g_prmanchor_thisnode_*.log"),
    ("PABH", "7B", 150, "bipace_7b_alfworld_bisimhead_g_prmanchor_seed{seed}_thisnode_*.log"),
]

# WebShop: (method, scale, budget, [filename glob patterns], seed)
WS_MANIFEST = [
    ("GRPO", "1.5B", 200, "grpo_1.5b_webshop_official_seed{seed}_node1613826_*.log"),
    ("GiGPO", "1.5B", 200, "gigpo_1.5b_webshop_official_seed{seed}_node1613826_*.log"),
    ("BiPACE-Q", "1.5B", 200, "bipace_1.5b_webshop_official_baseline_seed{seed}_node1613826_*.log"),
    ("PABH", "1.5B", 200, "bipace_1.5b_webshop_bisimhead_official_seed{seed}_node1613826_*.log"),
    ("BiPACE-Q", "7B", 150, "bipace_7b_webshop_baseline_seed{seed}_node1613826_*.log"),
    ("PABH", "7B", 150, "bipace_7b_webshop_bisimhead_seed{seed}_node1613826_*.log"),
    ("GiGPO", "7B", 150, "bipace_7b_webshop_gigpo_seed{seed}_node1613826_*.log"),
]


def parse_ws_log(path: str, budget: int):
    """Return (score_max, succ_max)."""
    score_max = -1.0
    succ_max = -1.0
    with open(path, errors="ignore") as fh:
        for raw in fh:
            line = raw.replace("\r", "\n").strip()
            m = re.search(r"step:(\d+) - ", line)
            if not m or int(m.group(1)) > budget:
                continue
            s = re.search(r"val/webshop_task_score \(not success_rate\):([0-9.]+)", line)
            if s:
                score_max = max(score_max, float(s.group(1)))
            u = re.search(r"val/success_rate:([0-9.]+)", line)
            if u:
                succ_max = max(succ_max, float(u.group(1)))
    return (score_max, succ_max) if score_max >= 0 else None


def log_patterns(entry):
    if len(entry) == 4:
        method, scale, budget, pat = entry
        return method, scale, budget, pat, None
    method, scale, budget, pat, seed0_pat = entry
    return method, scale, budget, pat, seed0_pat


def latest_log(pattern: str) -> str | None:
    files = sorted(glob.glob(os.path.join(LOG_DIR, pattern)))
    return files[-1] if files else None


def parse_alf_log(path: str, budget: int):
    """Return (peak_success, per_task_at_peak)."""
    steps = []
    with open(path, errors="ignore") as fh:
        for raw in fh:
            line = raw.replace("\r", "\n").strip()
            m = re.search(r"step:(\d+) - ", line)
            if not m:
                continue
            step = int(m.group(1))
            if step > budget:
                continue
            succ = re.search(r"val/success_rate:([0-9.]+)", line)
            if not succ:
                continue
            tasks = {}
            for name, key in ALF_TASKS:
                t = re.search(rf"val/{key}:([0-9.]+)", line)
                if t:
                    tasks[name] = float(t.group(1))
            steps.append((step, float(succ.group(1)), tasks))
    if not steps:
        return None
    peak_step, peak_succ, _ = max(steps, key=lambda s: s[1])
    peak_tasks = next((t for s, v, t in steps if s == peak_step), {})
    return peak_succ, peak_tasks


def main():
    result = {}
    print("=== ALFWorld per-task (peak val/success_rate step) ===")
    print(f"{'method':9s} {'scale':5s} {'seed':4s} {'All':>7s}  " +
          "  ".join(f"{n:>6s}" for n, _ in ALF_TASKS))
    for entry in ALF_MANIFEST:
        method, scale, budget, pattern, seed0_pattern = log_patterns(entry)
        for seed in range(6):
            p = pattern.format(seed=seed)
            log = latest_log(p)
            if not log and seed == 0 and seed0_pattern:
                log = latest_log(seed0_pattern)
            if not log:
                continue
            parsed = parse_alf_log(log, budget)
            if parsed is None:
                continue
            peak_succ, tasks = parsed
            key = f"{method} {scale} seed{seed}"
            result[key] = {"all": peak_succ, "tasks": tasks, "log": os.path.basename(log)}
            row = f"{method:9s} {scale:5s} s{seed:<3d} {peak_succ*100:6.1f}%  "
            row += "  ".join(f"{tasks.get(n, float('nan'))*100:6.1f}" for n, _ in ALF_TASKS)
            print(row)

    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(f"\nwrote {OUT}")

    # WebShop Score / Success
    ws_result = {}
    print("\n=== WebShop Score / Success (independently maxed) ===")
    print(f"{'method':9s} {'scale':5s} {'seed':4s} {'Score':>7s} {'Success':>8s}")
    for method, scale, budget, pattern in WS_MANIFEST:
        for seed in range(12):
            p = pattern.format(seed=seed)
            log = latest_log(p)
            if not log:
                continue
            parsed = parse_ws_log(log, budget)
            if parsed is None:
                continue
            score, succ = parsed
            key = f"{method} {scale} seed{seed}"
            ws_result[key] = {"score": score, "success": succ, "log": os.path.basename(log)}
            print(f"{method:9s} {scale:5s} s{seed:<3d} {score*100:6.1f}% {succ*100:7.1f}%")
    with open(OUT.replace("subtask", "webshop"), "w") as fh:
        json.dump(ws_result, fh, indent=2, sort_keys=True)
    print(f"\nwrote {OUT.replace('subtask', 'webshop')}")


if __name__ == "__main__":
    main()

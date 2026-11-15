# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    import gym
except ImportError:
    import gymnasium as gym


def load_sciworld_split(split_path: str, split_name: str) -> List[int | Dict[str, Any]]:
    """Load a ScienceWorld split from task_dev_sciworld.json-style files."""
    with open(os.path.expanduser(split_path), "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]

    if not isinstance(payload, dict) or split_name not in payload:
        raise ValueError(f"Cannot find split {split_name!r} in {split_path}")

    split_payload = payload[split_name]
    if isinstance(split_payload, dict) and "sciworld" in split_payload:
        split_payload = split_payload["sciworld"]
    if isinstance(split_payload, dict):
        split_payload = split_payload.get("idxs", split_payload.get("ids", split_payload.get("items")))
    if not isinstance(split_payload, list):
        raise ValueError(f"ScienceWorld split {split_name!r} must resolve to a list")
    return split_payload


def _extract_global_idx(record: int | str | Dict[str, Any]) -> int:
    if isinstance(record, (int, np.integer)):
        return int(record)
    if isinstance(record, str):
        match = re.search(r"(\d+)$", record)
        if match:
            return int(match.group(1))
        raise ValueError(f"Cannot parse ScienceWorld item id from {record!r}")
    if isinstance(record, dict):
        for key in ("global_idx", "idx", "id", "item_id", "variation_idx"):
            if key in record:
                return _extract_global_idx(record[key])
    raise ValueError(f"Unsupported ScienceWorld reset record: {record!r}")


def _resolve_task_name(task_names: Sequence[str], task_value: int | str) -> tuple[int, str]:
    if isinstance(task_value, (int, np.integer)):
        task_num = int(task_value)
        return task_num, task_names[task_num]
    if isinstance(task_value, str):
        if task_value.isdigit():
            task_num = int(task_value)
            return task_num, task_names[task_num]
        if task_value in task_names:
            return task_names.index(task_value), task_value
    raise ValueError(f"Cannot resolve ScienceWorld task from {task_value!r}")


def _build_global_idx_map(env: Any, task_names: Sequence[str]) -> List[tuple[int, str, int]]:
    idx_map: List[tuple[int, str, int]] = []
    for task_num, task_name in enumerate(task_names):
        max_variations = int(env.get_max_variations(task_name))
        idx_map.extend((task_num, task_name, variation_idx) for variation_idx in range(max_variations))
    return idx_map


def _resolve_record(
    env: Any,
    task_names: Sequence[str],
    global_idx_map: List[tuple[int, str, int]],
    record: int | str | Dict[str, Any],
) -> tuple[int, str, int, int | None]:
    if isinstance(record, dict):
        task_value = record.get("task_name", record.get("task_num", record.get("task_id")))
        variation_value = record.get("variation_idx", record.get("variation_id", record.get("variation")))
        if task_value is not None and variation_value is not None:
            task_num, task_name = _resolve_task_name(task_names, task_value)
            global_idx = record.get("global_idx", record.get("idx", record.get("item_id")))
            parsed_global_idx = _extract_global_idx(global_idx) if global_idx is not None else None
            return task_num, task_name, int(variation_value), parsed_global_idx

    global_idx = _extract_global_idx(record)
    if global_idx < 0 or global_idx >= len(global_idx_map):
        raise ValueError(
            f"ScienceWorld global_idx={global_idx} is outside the supported range [0, {len(global_idx_map)})"
        )
    task_num, task_name, variation_idx = global_idx_map[global_idx]
    return task_num, task_name, variation_idx, global_idx


def _make_env(jar_path: str | None, env_step_limit: int):
    from scienceworld import ScienceWorldEnv

    if jar_path:
        return ScienceWorldEnv("", os.path.expanduser(jar_path), envStepLimit=env_step_limit)
    return ScienceWorldEnv("", envStepLimit=env_step_limit)


def _format_available_actions(env: Any) -> str:
    actions = env.get_possible_actions()
    objects = env.get_possible_objects()
    return (
        f"Valid action templates: {actions}\n"
        f"Replace OBJ/LOC with one of the currently visible or relevant objects/locations: {objects}\n"
        "Example: <action>focus on door</action>"
    )


def _safe_possible_actions(env: Any) -> List[str]:
    try:
        return list(env.get_valid_action_object_combinations())
    except Exception:
        return []


def _score_from_info(info: Dict[str, Any]) -> float:
    score = info.get("score", 0.0)
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _compute_reward(info: Dict[str, Any], prev_score: float, reward_mode: str, reward_scale: float) -> float:
    if reward_mode == "score_delta":
        return max(_score_from_info(info) - prev_score, 0.0) * reward_scale / 100.0
    return reward_scale * float(info["won"])


def _compute_process_reward(won: bool, score_delta: float) -> float:
    """规则 PRM（锚专用，不进策略奖励流）：镜像 ALFWorld 事件 PRM 结构。

    win=+1.0（终局信号）+ 进度增量 max(Δscore,0)/100（稠密 0-100 官方 score）。
    无进展/无效动作自然得 0（score 不变）。供 bisim head 的 MICo 锚消费；
    仅在 env.sciworld.process_reward_mode='score_delta_prm' 时注入 infos，
    默认关闭（不改变任何既有行为）。
    """
    return (1.0 if won else 0.0) + max(score_delta, 0.0) / 100.0


def _worker(
    remote: mp.connection.Connection,
    seed: int,
    simplifications_preset: str,
    env_step_limit: int,
    jar_path: str | None,
    reward_mode: str,
    reward_scale: float,
    process_reward_mode: str = "",
):
    env = _make_env(jar_path, env_step_limit)
    task_names = env.get_task_names()
    global_idx_map = _build_global_idx_map(env, task_names)
    prev_score = 0.0
    current_task_info: Dict[str, Any] = {}

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == "reset":
                task_num, task_name, variation_idx, global_idx = _resolve_record(env, task_names, global_idx_map, data)
                simplification_str = simplifications_preset or ""
                env.load(task_name, variation_idx, simplification_str)
                observation, info = env.reset()

                info = dict(info or {})
                info["task_description"] = env.get_task_description()
                info["task_num"] = task_num
                info["task_name"] = task_name
                info["variation_idx"] = variation_idx
                info["global_idx"] = global_idx
                info["item_id"] = f"sciworld_{global_idx}" if global_idx is not None else None
                info["data_source"] = "sciworld"
                info["available_actions"] = _format_available_actions(env)
                info["possible_actions"] = _safe_possible_actions(env)
                info["observation_text"] = observation
                info["score"] = _score_from_info(info)
                info["task_score"] = info["score"]
                info["won"] = False
                if process_reward_mode == "score_delta_prm":
                    info["process_reward"] = 0.0
                current_task_info = {
                    "task_description": info["task_description"],
                    "task_num": task_num,
                    "task_name": task_name,
                    "variation_idx": variation_idx,
                    "global_idx": global_idx,
                    "item_id": info["item_id"],
                    "data_source": "sciworld",
                }
                prev_score = 0.0
                remote.send((observation, info))

            elif cmd == "step":
                observation, _, done, info = env.step(data)
                info = dict(info or {})
                current_score = _score_from_info(info)
                if current_score == -100:
                    current_score = prev_score
                    info["score"] = prev_score
                for key, value in current_task_info.items():
                    info.setdefault(key, value)

                info["task_score"] = current_score
                info["won"] = bool(done and current_score == 100.0)
                reward = _compute_reward(info, prev_score, reward_mode, reward_scale)
                if process_reward_mode == "score_delta_prm":
                    info["process_reward"] = _compute_process_reward(
                        info["won"], current_score - prev_score
                    )
                prev_score = current_score

                info["available_actions"] = _format_available_actions(env)
                info["possible_actions"] = _safe_possible_actions(env)
                info["observation_text"] = observation
                remote.send((observation, reward, done, info))

            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command sent to ScienceWorld worker: {cmd}")
    finally:
        env.close()


class SciWorldMultiProcessEnv(gym.Env):
    """Vectorized multiprocessing wrapper around ScienceWorldEnv."""

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        split_items: List[int | Dict[str, Any]] | None = None,
        simplifications_preset: str = "",
        env_step_limit: int = 100,
        jar_path: str | None = None,
        reward_mode: str = "terminal",
        reward_scale: float = 10.0,
        process_reward_mode: str = "",
    ) -> None:
        super().__init__()

        if not split_items:
            raise ValueError("ScienceWorld requires a non-empty split_items list")
        if not is_train and group_n != 1:
            raise ValueError("ScienceWorld validation env requires group_n=1")

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        self.split_items = list(split_items)
        self._rng = np.random.RandomState(seed)
        self.prev_available_actions = ["" for _ in range(self.num_processes)]
        self.prev_possible_actions = [[] for _ in range(self.num_processes)]
        self._closed = False

        self._parent_remotes: list[mp.connection.Connection] = []
        self._workers: list[mp.Process] = []
        ctx = mp.get_context("spawn")

        for i in range(self.num_processes):
            parent_remote, child_remote = ctx.Pipe()
            worker = ctx.Process(
                target=_worker,
                args=(
                    child_remote,
                    seed + (i // max(group_n, 1)),
                    simplifications_preset,
                    env_step_limit,
                    jar_path,
                    reward_mode,
                    reward_scale,
                    process_reward_mode,
                ),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
            self._parent_remotes.append(parent_remote)
            child_remote.close()

    def _sample_records(self) -> list[int | Dict[str, Any]]:
        replace = len(self.split_items) < self.env_num
        base_indices = self._rng.choice(len(self.split_items), size=self.env_num, replace=replace)
        records = [self.split_items[int(idx)] for idx in base_indices]
        return [record for record in records for _ in range(self.group_n)]

    def _normalize_kwargs(self, kwargs: Any) -> list[int | Dict[str, Any]]:
        if kwargs is None:
            return self._sample_records()
        if isinstance(kwargs, np.ndarray):
            kwargs = kwargs.tolist()
        records = list(kwargs)
        if len(records) == self.env_num and self.group_n > 1:
            return [record for record in records for _ in range(self.group_n)]
        if len(records) != self.num_processes:
            raise ValueError(
                f"Expected {self.num_processes} ScienceWorld reset records, got {len(records)}"
            )
        return records

    def reset(self, kwargs: Any = None):
        records = self._normalize_kwargs(kwargs)
        for remote, record in zip(self._parent_remotes, records):
            remote.send(("reset", record))

        obs_list, info_list = [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, info = remote.recv()
            obs_list.append(obs)
            info_list.append(info)
            self.prev_available_actions[i] = info["available_actions"]
            self.prev_possible_actions[i] = info["possible_actions"]
        return obs_list, info_list

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(f"Expected {self.num_processes} actions, got {len(actions)}")

        for remote, action in zip(self._parent_remotes, actions):
            remote.send(("step", action))

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, reward, done, info = remote.recv()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            self.prev_available_actions[i] = info["available_actions"]
            self.prev_possible_actions[i] = info["possible_actions"]
        return obs_list, reward_list, done_list, info_list

    @property
    def get_available_actions(self):
        return self.prev_available_actions

    @property
    def get_admissible_commands(self):
        return self.prev_available_actions

    @property
    def get_possible_actions(self):
        return self.prev_possible_actions

    def close(self):
        if self._closed:
            return
        for remote in self._parent_remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for worker in self._workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1)
        self._closed = True

    def __del__(self):
        self.close()


def build_sciworld_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    split_items: List[int | Dict[str, Any]] | None = None,
    simplifications_preset: str = "",
    env_step_limit: int = 100,
    jar_path: str | None = None,
    reward_mode: str = "terminal",
    reward_scale: float = 10.0,
    process_reward_mode: str = "",
):
    return SciWorldMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        split_items=split_items,
        simplifications_preset=simplifications_preset,
        env_step_limit=env_step_limit,
        jar_path=jar_path,
        reward_mode=reward_mode,
        reward_scale=reward_scale,
        process_reward_mode=process_reward_mode,
    )

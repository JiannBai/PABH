# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""ScienceWorld manager glue for verl-agent style multi-turn rollout.

This file is intentionally scoped to ScienceWorld. In a full verl-agent fork,
the same class can be inserted into ``agent_system/environments/env_manager.py``
and selected from ``make_envs(config)`` when ``env.env_name=ScienceWorld``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch

from .envs import build_sciworld_envs, load_sciworld_split
from .projection import sciworld_projection
from agent_system.environments.prompts.sciworld import (
    SCIWORLD_ACTIONS,
    SCIWORLD_TEMPLATE,
    SCIWORLD_TEMPLATE_NO_HIS,
    SCIWORLD_TEMPLATE_NO_HIS_TEMPLATE_ONLY,
    SCIWORLD_TEMPLATE_TEMPLATE_ONLY,
)


def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (int, float, bool, tuple, list)):
        return np.array(data)
    raise ValueError(f"Unsupported type: {type(data)}")


def _config_get(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class SciProgressTracker:
    """ScienceWorld 反馈事件解析器（g_text 生成，三层栈第 1 层）。

    覆盖 ScienceWorld 真实反馈的主动与被动两种语态，并为多参数动作保留
    任务关键的第二参数（目的地/容器/工具/读数）。与 ALFWorld 的
    AlfProgressTracker 同款语义：导航/无事件反馈不记录。
    """

    # (predicate, pattern) — 只记改变任务状态的操作事件；顺序即优先级。
    _EVENTS = [
        ("move",       r"You move the (?P<obj>[a-z0-9 ]+?) to the (?P<dest>[a-z0-9 ]+?)\."),
        ("pour",       r"You pour the (?P<obj>[a-z0-9 ]+?) into the (?P<dest>[a-z0-9 ]+?)\."),
        ("dunk",       r"You dunk the (?P<obj>[a-z0-9 ]+?) into the (?P<dest>[a-z0-9 ]+?)\."),
        ("mix",        r"You mix the (?P<obj>[a-z0-9 ]+?)\."),
        ("connect",    r"You connect the (?P<obj>[a-z0-9 ]+?) to the (?P<dest>[a-z0-9 ]+?)\."),
        ("disconnect", r"You disconnect the (?P<obj>[a-z0-9 ]+?)\."),
        ("heat",       r"You heat the (?P<obj>[a-z0-9 ]+?)(?: using the (?P<tool>[a-z0-9 ]+?))?\."),
        ("cool",       r"You cool the (?P<obj>[a-z0-9 ]+?)(?: using the (?P<tool>[a-z0-9 ]+?))?\."),
        ("use",        r"You use the (?P<obj>[a-z0-9 ]+?)(?: (?:on|with) the (?P<tool>[a-z0-9 ]+?))?\."),
        ("activate",   r"The (?P<obj>[a-z0-9 ]+?) (?:is now|is already) activated"),
        ("deactivate", r"The (?P<obj>[a-z0-9 ]+?) (?:is now|is already) deactivated"),
        ("open",       r"The (?P<obj>[a-z0-9 ]+?) (?:is now|is already) open"),
        ("close",      r"The (?P<obj>[a-z0-9 ]+?) (?:is now|is already) closed"),
        ("focus",      r"You focus on the (?P<obj>[a-z0-9 ]+?)\."),
        ("measure",    r"the (?P<obj>[a-z0-9 ]+?) measures a temperature of (?P<val>[0-9]+)"),
        ("read",       r"You read the (?P<obj>[a-z0-9 ]+?)\."),
    ]

    def __init__(self):
        self.events: list[tuple[str, str, str]] = []  # 有序去重的 (pred, obj, arg)

    def update(self, obs_text: str) -> None:
        import re
        # 只取反馈正文第一行，避免房间描述被误判为事件。
        first_line = (obs_text or "").strip().split("\n")[0].strip()
        if not first_line:
            return
        for pred, pat in self._EVENTS:
            m = re.search(pat, first_line, re.IGNORECASE)
            if m:
                gd = m.groupdict()
                obj = (gd.get("obj") or "").strip()
                arg = (gd.get("dest") or gd.get("tool") or gd.get("val") or "").strip()
                key = (pred, obj, arg)
                if key not in self.events:
                    self.events.append(key)
                break

    def g_text(self) -> str:
        if not self.events:
            return ""
        evs = ", ".join(f"{p}({o}" + (f"→{a})" if a else ")") for p, o, a in self.events)
        return f"[Completed subgoal events: {evs}]"


class SimpleMemory:
    """Small per-environment history buffer used by prompt construction."""

    def __init__(self):
        self._items: list[list[dict[str, str]]] = []

    def reset(self, batch_size: int):
        self._items = [[] for _ in range(batch_size)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index: int):
        return self._items[index]

    def store(self, payload: dict[str, list[str]]):
        text_obs = payload["text_obs"]
        actions = payload["action"]
        for idx, (obs, action) in enumerate(zip(text_obs, actions)):
            self._items[idx].append({"text_obs": obs, "action": action})

    def fetch(self, history_length: int, obs_key: str, action_key: str):
        contexts, valid_lens = [], []
        for history in self._items:
            recent = history[-history_length:] if history_length >= 0 else history
            start = len(history) - len(recent)
            lines = []
            for offset, record in enumerate(recent):
                step = start + offset + 1
                lines.append(
                    f"[Observation {step}: '{record[obs_key]}', "
                    f"Action {step}: '{record[action_key]}']"
                )
            contexts.append("\n".join(lines))
            valid_lens.append(len(recent))
        return contexts, valid_lens


class SciWorldEnvironmentManager:
    """Prompt/action/reward adapter between verl-agent rollout and ScienceWorld."""

    def __init__(self, envs, projection_f=sciworld_projection, config=None):
        self.envs = envs
        self.projection_f = projection_f
        self.config = config
        self.memory = SimpleMemory()
        self.tasks: list[str] = []
        self.pre_text_obs: list[str] = []
        # ---- g_text 注入（三层栈第 1 层，与 ALFWorld/WebShop 同款 gate 语义）----
        # 注入条件：bisim_embedder == "bisim_head"（我们的方法臂）或显式 env.sciworld.gtext_inject
        # baseline/GiGPO 等对比臂（actor_hidden / hash_ngram / 不传）一律不注入 —— anchor == 纯 obs
        self._inject_g_text = False
        try:
            embedder = str(getattr(config.algorithm.gigpo, "bisim_embedder", "actor_hidden"))
            _env_cfg = _config_get(config, "env", None)
            explicit = bool(_config_get(_env_cfg, "gtext_inject", False)) if _env_cfg is not None else False
            self._inject_g_text = (embedder == "bisim_head") or explicit
        except Exception:
            self._inject_g_text = False
        self._progress_trackers = [SciProgressTracker() for _ in range(getattr(envs, "num_processes", 0) or 0)]

    def _g_texts(self, text_obs: List[str]) -> List[str]:
        """每步解析反馈事件，累积进度集，返回 goal-conditioned g_text 列表。"""
        out = []
        n = len(text_obs)
        while len(self._progress_trackers) < n:
            self._progress_trackers.append(SciProgressTracker())
        for i in range(n):
            self._progress_trackers[i].update(text_obs[i])
            ev = self._progress_trackers[i].g_text()
            goal = self.tasks[i] if i < len(self.tasks) else ""
            if goal and ev:
                out.append(f"[Task: {goal}]\n{ev}")
            elif goal:
                out.append(f"[Task: {goal}]")
            else:
                out.append(ev)
        return out

    def _anchors(self, text_obs: List[str]) -> List[str]:
        if self._inject_g_text:
            gs = self._g_texts(text_obs)
            return [f"{g}\n\n{o}" if g else o for g, o in zip(gs, text_obs)]
        return list(text_obs)

    def reset(self, kwargs):
        text_obs, infos = self.envs.reset(kwargs=kwargs)
        self.memory.reset(batch_size=len(text_obs))
        self.pre_text_obs = text_obs
        self.tasks = [info.get("task_description", "Unknown task") for info in infos]
        self._progress_trackers = [SciProgressTracker() for _ in range(len(text_obs))]

        observations = {
            "text": self.build_text_obs(text_obs, infos, init=True),
            "image": None,
            "anchor": self._anchors(text_obs),
        }
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        self.pre_text_obs = text_obs

        next_observations = {
            "text": self.build_text_obs(text_obs, infos),
            "image": None,
            "anchor": self._anchors(text_obs),
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = _to_numpy(valids[i])
            info["score"] = info.get("score", info.get("task_score", 0.0))
            info["raw_response"] = text_actions[i]
            info["parsed_action"] = actions[i]

        return next_observations, _to_numpy(rewards), _to_numpy(dones), infos

    def build_text_obs(self, text_obs: List[str], infos: List[Dict], init: bool = False) -> List[str]:
        postprocess_text_obs = []
        history_length = _config_get(_config_get(self.config, "env", None), "history_length", 2)

        sciworld_config = None
        if self.config is not None and hasattr(self.config, "env") and hasattr(self.config.env, "sciworld"):
            sciworld_config = self.config.env.sciworld
        prompt_mode = _config_get(sciworld_config, "prompt_mode", "with_hints")

        if prompt_mode in ("with_hints", "default"):
            template_no_his = SCIWORLD_TEMPLATE_NO_HIS
            template_with_his = SCIWORLD_TEMPLATE
        elif prompt_mode in ("template_only", "no_hints"):
            template_no_his = SCIWORLD_TEMPLATE_NO_HIS_TEMPLATE_ONLY
            template_with_his = SCIWORLD_TEMPLATE_TEMPLATE_ONLY
        else:
            raise ValueError(f"Unsupported ScienceWorld prompt_mode: {prompt_mode!r}")

        if not init and history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                history_length,
                obs_key="text_obs",
                action_key="action",
            )

        for i in range(len(text_obs)):
            available_actions = infos[i].get("available_actions", "")
            if init or history_length <= 0:
                obs = template_no_his.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    general_actions=SCIWORLD_ACTIONS,
                    available_actions=available_actions,
                )
            else:
                obs = template_with_his.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    general_actions=SCIWORLD_ACTIONS,
                    available_actions=available_actions,
                )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        success = defaultdict(list)

        for batch_idx in range(len(total_batch_list)):
            for i in reversed(range(len(total_batch_list[batch_idx]))):
                batch_item = total_batch_list[batch_idx][i]
                if batch_item["active_masks"]:
                    info = total_infos[batch_idx][i]
                    success["success_rate"].append(float(info["won"]))
                    success["sciworld_score"].append(float(info.get("task_score", 0.0)))
                    break

        return {key: np.array(value) for key, value in success.items()}

    def close(self):
        self.envs.close()


def make_sciworld_envs(config):
    """Create train/validation ScienceWorld managers from a verl-agent config."""

    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    sciworld_config = _config_get(config.env, "sciworld", {})
    split_path = _config_get(sciworld_config, "split_path", "task_dev_sciworld.json")
    train_split = _config_get(sciworld_config, "train_split", "rl")
    val_split = _config_get(sciworld_config, "val_split", "test")
    train_items = load_sciworld_split(split_path, train_split)
    val_items = load_sciworld_split(split_path, val_split)

    env_step_limit = _config_get(sciworld_config, "env_step_limit", None)
    if env_step_limit is None:
        env_step_limit = config.env.max_steps

    env_kwargs = {
        "simplifications_preset": _config_get(sciworld_config, "simplifications_preset", ""),
        "env_step_limit": env_step_limit,
        "jar_path": _config_get(sciworld_config, "jar_path", None),
        "reward_mode": _config_get(sciworld_config, "reward_mode", "terminal"),
        "reward_scale": _config_get(sciworld_config, "reward_scale", 10.0),
        # 规则 PRM 锚（bisim head MICo 监督用，不进策略奖励流）；
        # 默认 ''=关闭（既有行为），'score_delta_prm' 时 infos 暴露 process_reward
        "process_reward_mode": _config_get(sciworld_config, "process_reward_mode", ""),
    }

    train_env = build_sciworld_envs(
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        is_train=True,
        split_items=train_items,
        **env_kwargs,
    )
    val_env = build_sciworld_envs(
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=1,
        is_train=False,
        split_items=val_items,
        **env_kwargs,
    )
    return SciWorldEnvironmentManager(train_env, sciworld_projection, config), SciWorldEnvironmentManager(
        val_env, sciworld_projection, config
    )

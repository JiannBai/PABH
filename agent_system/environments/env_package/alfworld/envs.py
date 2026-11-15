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

import os
import sys
import time
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray


def _ray_get_with_progress(futures, op_name, timeout_s=None, heartbeat_s=30):
    """Wait for futures with optional timeout + per-30s heartbeat.

    Why: bare ray.get(futures) blocks forever if any worker dies during
    init/reset/step (e.g. TextWorld OOM, missing data file). On ALFWorld
    runs with 128+ env actors, a silent stuck future stalls the whole
    driver with no log output for hours.
    """
    if timeout_s is None:
        timeout_s = float(os.environ.get("ALFWORLD_RAY_TIMEOUT", "1800"))
    pending = list(futures)
    done = []
    deadline = time.monotonic() + timeout_s
    last_beat = time.monotonic()
    while pending:
        ready, pending = ray.wait(pending, num_returns=len(pending), timeout=heartbeat_s)
        done.extend(ready)
        now = time.monotonic()
        if now - last_beat >= heartbeat_s and pending:
            print(f"[AlfworldEnvs] {op_name}: {len(done)}/{len(futures)} done, "
                  f"{len(pending)} pending after {int(now - (deadline - timeout_s))}s",
                  flush=True)
            last_beat = now
        if now > deadline and pending:
            raise TimeoutError(
                f"[AlfworldEnvs] {op_name} timed out: {len(pending)}/{len(futures)} "
                f"workers stuck after {timeout_s:.0f}s. "
                "Set ALFWORLD_RAY_TIMEOUT to override.")
    return ray.get(done)

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]

# ---------------------------------------------------------------------------
# Progress-state tracking (g_set / g_text) for the progress-aware bisim head.
#
# ALFWorld feedback text is template-generated and closed-set; every
# environment-changing event maps to exactly one of six sentence templates:
#   "You pick up the {obj} from the {receptacle}."
#   "You clean the {obj} using the {receptacle}."
#   "You cool the {obj} using the {receptacle}."
#   "You heat the {obj} using the {receptacle}."
#   "You move the {obj} to the {receptacle}."        (== put)
#   "You turn on the {receptacle}."                  (== use desklamp)
# Navigation/open/close feedback ("You arrive at ...", "You open ...") carries
# no task-progress content and is deliberately NOT part of g_set: two states
# that differ only by the agent's location are still bisimilar w.r.t. the
# crafting-free household goals (the action feedback string in the anchor obs
# already covers them). Keeping g_set event-only also makes it invariant to
# detours, matching the TextCraft `_get_g_set` semantics.
# ---------------------------------------------------------------------------

import re as _re

_ALF_EVENT_PATTERNS = [
    # (predicate, compiled pattern with named groups)
    ("take", _re.compile(r"^You pick up the (?P<obj>[a-z0-9 ]+?) from the (?P<rec>[a-z0-9 ]+?)\.")),
    ("clean", _re.compile(r"^You clean the (?P<obj>[a-z0-9 ]+?) using the (?P<rec>[a-z0-9 ]+?)\.")),
    ("cool", _re.compile(r"^You cool the (?P<obj>[a-z0-9 ]+?) using the (?P<rec>[a-z0-9 ]+?)\.")),
    ("heat", _re.compile(r"^You heat the (?P<obj>[a-z0-9 ]+?) using the (?P<rec>[a-z0-9 ]+?)\.")),
    ("put", _re.compile(r"^You move the (?P<obj>[a-z0-9 ]+?) to the (?P<rec>[a-z0-9 ]+?)\.")),
    ("toggle", _re.compile(r"^You turn on the (?P<rec>[a-z0-9 ]+?)\.")),
]


class AlfProgressTracker:
    """Per-episode cumulative progress-set tracker.

    g_set elements are (predicate, object) tuples, e.g.
        {("take","egg 3"), ("heat","egg 3"), ("put","egg 3")}
    Receptacle/device arguments are dropped from the key so that the same
    subgoal accomplished with different appliances (fridge 1 vs fridge 2)
    maps to the same progress element; object identity (with its trailing
    index, e.g. "egg 3") is kept because pick_two-style tasks distinguish
    kettle 1 vs kettle 3.
    """

    def __init__(self):
        self.events = []  # ordered list of (pred, obj) for g_text

    def update(self, obs_text: str) -> None:
        """Parse one feedback line and accumulate any event into the set."""
        first_line = obs_text.strip().split("\n")[0].strip()
        for pred, pat in _ALF_EVENT_PATTERNS:
            m = pat.match(first_line)
            if m:
                obj = m.groupdict().get("obj")
                key = (pred, obj if obj else m.group("rec"))
                if key not in self.events:
                    self.events.append(key)
                break

    def g_set(self):
        return frozenset(self.events)

    def g_text(self) -> str:
        if not self.events:
            return ""
        done = ", ".join(f"{p}({o})" for p, o in self.events)
        return f"[Progress] completed subgoals: {done}"


def _alf_g_info(tracker: AlfProgressTracker, won) -> dict:
    info = {"g_set": tracker.g_set(), "g_text": tracker.g_text()}
    if won is not None:
        # Dense-ish scalar progress for logging/diagnostics only (not used as
        # the MICo anchor — the structured g_set via Jaccard is).
        info["g_progress_frac"] = 1.0 if won else None
    return info

# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds one sub-environment
        self.env.seed(seed)
        self.progress = AlfProgressTracker()

    def step(self, action):
        """Execute a step in the environment"""
        actions = [action]

        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        # Progress-state tracking (env-agnostic consumers read info keys).
        n_before = len(self.progress.events)
        self.progress.update(obs[0])
        won = None
        try:
            won = bool(infos['won'][0]) if isinstance(infos.get('won'), (list, tuple)) else infos.get('won')
        except (KeyError, IndexError, TypeError):
            pass
        # Rule-based PRM (TextCraft-compatible semantics): +1 win, +0.5 per
        # new subgoal event, -0.1 failed/no-op action, 0 navigation. In
        # bisim_anchor_mode='d_g' the |r_i - r_j| term is replaced by the
        # dense Jaccard anchor; this value only feeds the legacy 'prm' path
        # and the episode reward logger.
        if won:
            process_reward = 1.0
        elif len(self.progress.events) > n_before:
            process_reward = 0.5
        elif "Nothing happens." in obs[0]:
            process_reward = -0.1
        else:
            process_reward = 0.0
        infos['process_reward'] = process_reward
        infos.update(_alf_g_info(self.progress, won))
        return obs, scores, dones, infos

    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        self.progress = AlfProgressTracker()
        infos['process_reward'] = 0.0
        infos.update(_alf_g_info(self.progress, None))
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        split = 'train' if is_train else eval_dataset
        t0 = time.monotonic()
        print(f"[AlfworldEnvs] building base_env (split={split}, env_num={env_num}, group_n={group_n})…", flush=True)
        base_env = get_environment(env_type)(config, train_eval=split)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Put base_env into Ray's plasma store ONCE so all workers share a
        # single read-only copy instead of pickling it per-actor (saves
        # GBs of pickle traffic + driver memory spike for N≥128 actors).
        base_env_ref = ray.put(base_env)
        print(f"[AlfworldEnvs] base_env ready in {time.monotonic()-t0:.1f}s; "
              f"spawning {self.num_processes} actors (lazy game init on first reset)…",
              flush=True)

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env_ref)
            self.workers.append(worker)
        print(f"[AlfworldEnvs] {self.num_processes} actor handles spawned (split={split})", flush=True)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = _ray_get_with_progress(futures, "step")
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                # Unwrap the per-worker batch dim; scalars/objects added by
                # the worker (process_reward, g_set, g_text) pass through.
                if isinstance(info[k], (list, tuple)):
                    info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        t0 = time.monotonic()
        print(f"[AlfworldEnvs] reset() dispatched to {len(self.workers)} workers "
              "(first call triggers lazy TextWorld game init)…", flush=True)
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results — bounded wait + per-30s heartbeat protects against
        # a silently-stuck worker hanging the whole driver.
        results = _ray_get_with_progress(futures, "reset")
        print(f"[AlfworldEnvs] reset() complete in {time.monotonic()-t0:.1f}s", flush=True)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                # Unwrap the per-worker batch dim; scalars/objects added by
                # the worker (process_reward, g_set, g_text) pass through.
                if isinstance(info[k], (list, tuple)):
                    info[k] = info[k][0]
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = _ray_get_with_progress(futures, "getobs")
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
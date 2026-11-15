# BiPACE modifications Copyright 2026 BiPACE contributors.
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
import time

import numpy as np
import ray
import requests

# ---- Thin HTTP client for the AppWorld environment server ----
# appworld 0.1.3 requires pydantic<2, which conflicts with vLLM 0.11
# (pydantic>=2.11.7) in the training env.  The standalone AppWorld server runs
# in a separate pydantic-v1 venv; the training-side worker talks to it over
# HTTP instead of importing appworld directly.

_APPWORLD_DATA_ROOT = os.environ.get("APPWORLD_DATA_ROOT", "/mnt/tidalfs-hssh01/data")


def load_task_ids(dataset_name):
    path = os.path.join(_APPWORLD_DATA_ROOT, "datasets", f"{dataset_name}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"AppWorld task id file not found: {path}")
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


class _ThinTask:
    def __init__(self, instruction, supervisor):
        self.instruction = instruction
        self.supervisor = supervisor


class _ThinEvalResult:
    def __init__(self, success):
        self.success = success


class AppWorld:
    """Minimal drop-in replacement for appworld.AppWorld in remote mode."""

    def __init__(self, task_id, experiment_name=None, remote_environment_url=None, max_interactions=1000):
        self.task_id = str(task_id)
        self._base = (remote_environment_url or "").rstrip("/")
        payload = {
            "task_id": self.task_id,
            "experiment_name": experiment_name or "default",
            "max_interactions": int(max_interactions),
        }
        r = requests.post(f"{self._base}/initialize", json=payload, timeout=60)
        r.raise_for_status()
        out = r.json()["output"]
        self.task = _ThinTask(out["instruction"], out.get("supervisor", {}))

    def execute(self, code):
        r = requests.post(
            f"{self._base}/execute",
            json={"task_id": self.task_id, "code": code},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["output"]

    def task_completed(self):
        r = requests.post(
            f"{self._base}/task_completed",
            json={"task_id": self.task_id},
            timeout=60,
        )
        r.raise_for_status()
        return bool(r.json()["output"])

    def evaluate(self, suppress_errors=True):
        r = requests.post(
            f"{self._base}/evaluate",
            json={"task_id": self.task_id, "suppress_errors": bool(suppress_errors), "report": False},
            timeout=120,
        )
        r.raise_for_status()
        out = r.json()["output"]
        return _ThinEvalResult(bool(out.get("success", False)))

    def close(self):
        try:
            requests.post(f"{self._base}/close", json={"task_id": self.task_id}, timeout=10)
        except Exception:
            pass


def load_available_ports(port_file="appworld_ports.ports"):
    """
    Load available port list from file
    """
    if not os.path.exists(port_file):
        raise FileNotFoundError(f"Port file {port_file} does not exist. Please run the service startup script first.")

    ports = []
    with open(port_file) as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                ports.append(int(line))

    if not ports:
        raise ValueError(f"No valid ports found in port file {port_file}.")

    return ports


class AppWorldWorker:
    """
    Ray Actor that holds an instance of AppWorld and operates the environment
    based on method calls from the main process.

    进度感知扩展（三层栈第 1/2 层数据源，2026-10-15，additive 默认关）：
    - g_set：episode 内成功执行过的 (app, api) 调用对（API 调用即任务进度）
    - g_text：累计 API 调用文本，由 manager 按 gate 注入 anchor
    - process_reward：win(+1.0) + 0.1×新 API 调用（进度增量） − 0.1×无效动作
      仅当 process_reward_mode='api_prm' 时暴露（镜像 ALFWorld 事件 PRM 结构）
    """

    def __init__(self, worker_id, max_interactions, port, process_reward_mode=""):
        self.env = None
        self.current_step_count = 0
        self.max_interactions = max_interactions
        self.worker_id = worker_id
        # Cast to str: task_ids may arrive as numpy str_ from np.random.choice; AppWorld is picky.
        self.url = f"http://0.0.0.0:{port}"
        self._prm_mode = process_reward_mode
        self._api_calls = []  # (app, api) 有序去重

    def _update_progress(self, action: str, obs: str):
        """解析动作代码中的 apis.<app>.<api>(...) 调用，记入进度集。
        以动作代码为准（而非输出）：写了调用即代表策略推进到该 API（成功率语义
        后续由 obs 的报错信息过滤——报错行不计数）。"""
        import re as _re
        if not self._prm_mode:
            return
        for m in _re.finditer(r"apis\.([a-z_]+)\.([a-z_]+)\s*\(", str(action)):
            key = (m.group(1), m.group(2))
            if key not in self._api_calls:
                # 排除元 API（文档查询类，非任务进度）
                if m.group(2) not in ("show_api_doc", "show_api_descriptions", "show_app_descriptions"):
                    self._api_calls.append(key)

    def _g_text(self) -> str:
        if not self._api_calls:
            return ""
        evs = ", ".join(f"{a}.{b}" for a, b in self._api_calls[-12:])
        return f"[Completed API calls: {evs}]"

    def _process_reward(self, won: bool, prev_calls: int, action_valid: bool) -> float:
        r = 1.0 if won else 0.0
        r += 0.1 * max(len(self._api_calls) - prev_calls, 0)
        if not action_valid:
            r -= 0.1
        return r

    def reset(self, task_id):
        """Reset the environment with a new task."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass

        self.current_step_count = 0
        self._api_calls = []

        self.env = AppWorld(
            task_id=str(task_id),
            experiment_name=f"default_{self.worker_id}",
            remote_environment_url=self.url,
        )

        obs = self.env.task.instruction
        info = {
            "task_id": str(task_id),
            "supervisor": dict(self.env.task.supervisor),
        }
        if self._prm_mode:
            info["g_set"] = list(self._api_calls)
            info["g_text"] = self._g_text()
            info["process_reward"] = 0.0
        return obs, info

    def step(self, action):
        """Execute one step in the environment."""
        if self.env is None:
            raise RuntimeError("Environment not reset before step. Please call reset().")

        self.current_step_count += 1
        prev_calls = len(self._api_calls)

        obs = self.env.execute(action)

        done = self.env.task_completed() or (self.current_step_count >= self.max_interactions)

        self._update_progress(action, obs)

        if done:
            is_success = self.env.evaluate().success

            reward = 10.0 if is_success else 0.0
            info = {"won": is_success, "step_count": self.current_step_count}
        else:
            reward = 0.0
            info = {"won": False, "step_count": self.current_step_count}

        if self._prm_mode:
            # 动作有效性：无 <code> 标签或代码为空（projection 侧会再判，此处粗判）
            action_valid = "<code>" in str(action) and "</code>" in str(action)
            info["g_set"] = list(self._api_calls)
            info["g_text"] = self._g_text()
            info["process_reward"] = self._process_reward(
                info["won"], prev_calls, action_valid
            )

        return obs, reward, done, info

    def close(self):
        """Close the environment."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass


class AppWorldEnvs:
    """
    A Ray-based distributed wrapper for AppWorld.
    - Creates multiple Ray actors, each holding a separate AppWorld instance.
    - Implements Gym-style interfaces such as step() / reset() / close().
    """

    def __init__(
        self,
        dataset_name,
        max_interactions,
        seed,
        env_num,
        group_n,
        start_server_id,
        resources_per_worker,
        port_file=None,
        is_train=True,
        step_timeout_s=180.0,
        process_reward_mode="",
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.max_interactions = max_interactions
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.task_ids = list(load_task_ids(dataset_name))
        self.is_train = is_train
        self.step_timeout_s = float(step_timeout_s)
        self._prm_mode = process_reward_mode

        # Per-instance RNG seeded explicitly; previously seed was accepted but never used,
        # so train/val task sampling was non-reproducible across runs.
        self.rng = np.random.default_rng(seed)
        # Deterministic cursor for eval: walk through task_ids in fixed order so val metrics
        # are comparable across checkpoints.
        self._eval_cursor = 0

        if self.env_num > len(self.task_ids):
            raise ValueError(f"Env_num ({self.env_num}) exceeds available task_ids in '{self.dataset_name}' ({len(self.task_ids)}). Please reducing env_num to {len(self.task_ids)}.")

        # Port file: allow absolute path override via env var, fall back to CWD-relative default.
        if port_file is None:
            port_file = os.environ.get("APPWORLD_PORT_FILE", "appworld_ports.ports")
        all_ports = load_available_ports(port_file)

        self.available_ports = all_ports[start_server_id : start_server_id + self.num_processes]

        # Check if we have enough ports
        if len(self.available_ports) < self.num_processes:
            raise ValueError(f"Need {self.num_processes} ports starting at index {start_server_id}, but only {len(self.available_ports)} ports available in '{port_file}'. Please ensure enough service instances are started.")

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Create Ray actors (workers)
        self._env_worker_cls = ray.remote(**resources_per_worker)(AppWorldWorker)
        self._resources_per_worker = resources_per_worker
        self._start_server_id = start_server_id
        self.workers = []
        for i in range(self.num_processes):
            self.workers.append(self._build_worker(i))

    def _build_worker(self, slot_idx):
        port = self.available_ports[slot_idx]
        return self._env_worker_cls.remote(
            worker_id=self._start_server_id + slot_idx,
            max_interactions=self.max_interactions,
            port=port,
            process_reward_mode=self._prm_mode,
        )

    def _gather_with_timeout(self, futures, on_timeout):
        """Resolve a list of futures with a per-call timeout.

        Workers whose call exceeds self.step_timeout_s are killed and replaced; their
        slot gets the value produced by on_timeout(slot_idx). This prevents a stuck
        AppWorld backend from hanging the whole training step.
        """
        results = [None] * len(futures)
        pending = {f: i for i, f in enumerate(futures)}
        deadline = time.time() + self.step_timeout_s
        while pending:
            remaining = max(0.0, deadline - time.time())
            done_fs, _ = ray.wait(list(pending.keys()), num_returns=len(pending), timeout=remaining)
            if not done_fs:
                # Hit deadline: anything still pending is considered hung.
                for f, idx in list(pending.items()):
                    try:
                        ray.cancel(f, force=True)
                    except Exception:
                        pass
                    try:
                        ray.kill(self.workers[idx])
                    except Exception:
                        pass
                    self.workers[idx] = self._build_worker(idx)
                    results[idx] = on_timeout(idx)
                pending.clear()
                break
            for f in done_fs:
                idx = pending.pop(f)
                try:
                    results[idx] = ray.get(f)
                except Exception as e:
                    # Worker raised — surface a synthetic terminal step so training can advance.
                    try:
                        ray.kill(self.workers[idx])
                    except Exception:
                        pass
                    self.workers[idx] = self._build_worker(idx)
                    results[idx] = on_timeout(idx, exc=e)
        return results

    def step(self, actions):
        """
        actions: Must be a list with length equal to self.num_processes,
        each sent to the corresponding worker.

        Return format follows Gym's step() convention:
            observations, rewards, dones, infos
        """
        assert len(actions) == self.num_processes, "The length of actions must match the number of processes."

        futures = [w.step.remote(actions[i]) for i, w in enumerate(self.workers)]

        def _on_step_timeout(slot_idx, exc=None):
            return (
                "[ENV ERROR: step timed out or worker crashed]",
                0.0,
                True,
                {"won": False, "step_count": -1, "timeout": True, "error": str(exc) if exc else "timeout"},
            )

        results = self._gather_with_timeout(futures, _on_step_timeout)

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Reset all worker environments simultaneously,
        returning each environment's initial observation and info.

        Train mode: sample env_num distinct tasks (seeded) and replicate group_n times so each
        group sees the same task — required for GRPO/GiGPO group-relative advantage.
        Eval mode: walk task_ids in fixed order; group_n is expected to be 1.
        """
        if self.is_train:
            task_id = self.rng.choice(self.task_ids, self.env_num, replace=False)
            task_id = np.repeat(task_id, self.group_n).tolist()
        else:
            picked = []
            for _ in range(self.env_num):
                picked.append(self.task_ids[self._eval_cursor % len(self.task_ids)])
                self._eval_cursor += 1
            task_id = np.repeat(np.array(picked, dtype=object), self.group_n).tolist()

        futures = [w.reset.remote(task_id[i]) for i, w in enumerate(self.workers)]

        def _on_reset_timeout(slot_idx, exc=None):
            return (
                "[ENV ERROR: reset timed out or worker crashed]",
                {"task_id": task_id[slot_idx], "supervisor": {"first_name": "", "last_name": "", "email": "", "phone_number": ""}, "timeout": True, "error": str(exc) if exc else "timeout"},
            )

        results = self._gather_with_timeout(futures, _on_reset_timeout)

        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def close(self):
        """Close all workers."""
        # Send close commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.close.remote()
            futures.append(future)

        # Wait for all workers to close
        ray.get(futures)

        # Shutdown Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def render(self):
        """Implement this if visualization is needed."""
        pass


def build_appworld_envs(
    dataset_name="train",
    max_interactions=50,
    seed=0,
    env_num=1,
    group_n=1,
    start_server_id=0,
    resources_per_worker=None,
    is_train=True,
    step_timeout_s=180.0,
    process_reward_mode="",
):
    if resources_per_worker is None:
        resources_per_worker = {"num_cpus": 0.1}
    return AppWorldEnvs(
        dataset_name=dataset_name,
        max_interactions=max_interactions,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        start_server_id=start_server_id,
        resources_per_worker=resources_per_worker,
        process_reward_mode=process_reward_mode,
        is_train=is_train,
        step_timeout_s=step_timeout_s,
    )

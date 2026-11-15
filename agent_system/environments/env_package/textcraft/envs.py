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
"""Ray-vectorised wrapper around the TextCraft env.

* one Ray actor per env slot, reset() takes an explicit goal_item so the parent
  process owns the train/val split (no global RNG dependence).
* train mode samples goals from the committed train split with the seeded RNG.
* val mode walks a shuffled val split round-robin.
* info dict surfaces `won`, `depth`, `goal_item` so downstream metrics can
  stratify success-rate by recipe depth (the entire reason this env exists for
  our CACB/GiGPO horizon ablation).
"""

import json
import os
import random
from typing import Dict, List

import gymnasium as gym
import numpy as np
import ray

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DATA_DIR = os.path.join(_HERE, "textcraft_src", "data")
_SPLIT_PATH = os.path.join(_HERE, "textcraft_splits.json")
_GOAL_INDEX_PATH = os.path.join(_HERE, "textcraft_goal_index.json")


def _load_goal_index():
    """Return committed depth annotations derived from the bundled recipes."""
    if not os.path.exists(_GOAL_INDEX_PATH):
        raise FileNotFoundError(f"Bundled TextCraft goal index not found at {_GOAL_INDEX_PATH}. Restore it from the BiPACE repository checkout; prepare_textcraft.py consumes this file but does not generate it.")
    with open(_GOAL_INDEX_PATH) as f:
        return {row["goal_item"]: int(row["depth"]) for row in json.load(f)}


def _load_split(is_train: bool) -> List[str]:
    """Return list of goal item ids belonging to the train or val split."""
    if not os.path.exists(_SPLIT_PATH):
        raise FileNotFoundError(f"Bundled TextCraft split not found at {_SPLIT_PATH}. Restore it from the BiPACE repository checkout; prepare_textcraft.py consumes this file but does not generate it.")
    with open(_SPLIT_PATH) as f:
        data = json.load(f)
    return list(data["train"] if is_train else data["val"])


# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------


class TextCraftWorker:
    """One TextCraft env per Ray actor.

    `reset(goal_item, depth)` bypasses the upstream seed-based goal lookup so the
    parent process can drive the train/val split deterministically.
    """

    def __init__(self, seed: int, env_kwargs: dict):
        # Lazy import — keeps the parent process free of gymnasium / textcraft state.
        from .textcraft_src.env import TextCraft

        # The vendored env.__init__ accepts a minecraft_dir that contains recipes/.
        self._env = TextCraft(minecraft_dir=_VENDOR_DATA_DIR)
        self._rng = random.Random(seed)
        self._max_steps = int(env_kwargs.get("max_steps", 30))
        self._max_distractor = int(env_kwargs.get("max_distractor", 10))
        self._reward_scale = float(env_kwargs.get("reward_scale", 10.0))
        # If True (default), the obs at reset only contains the goal's transitive
        # recipe closure (+ a few distractors); if False, fall back to upstream
        # behaviour (still goal-closure but un-trimmed).
        self._show_only_relevant_recipes = bool(env_kwargs.get("show_only_relevant_recipes", True))
        self._step_count = 0
        self._goal_item = None
        self._goal_depth = None
        self._closure_items = set()
        self._closure_inputs = set()

    # -- helpers ---------------------------------------------------------------

    def _build_initial_observation(self) -> str:
        """Replicate upstream reset()'s recipe-set assembly but with our own RNG and
        goal already fixed on self._env.
        """
        from .textcraft_src.utils import item_id_to_str

        env = self._env
        recipes, distractors = env.crafting_tree.create_recipe_set(env.goal)
        recipes_set = {r.recipe_str for r in recipes}
        distractor_set = set()
        for d in distractors:
            if d.recipe_str not in recipes_set:
                distractor_set.add(d.recipe_str)

        recipes_list = list(recipes_set)
        if self._show_only_relevant_recipes:
            n_dist = min(len(distractor_set), self._max_distractor)
        else:
            n_dist = len(distractor_set)
        if n_dist > 0:
            recipes_list += self._rng.sample(list(distractor_set), n_dist)
        self._rng.shuffle(recipes_list)
        return "Crafting commands:\n{}\n\nGoal: craft {}.".format("\n".join(recipes_list), item_id_to_str(env.goal))

    # -- API -------------------------------------------------------------------

    def _inventory_str(self) -> str:
        from .textcraft_src.utils import item_id_to_str

        inv = self._env.inventory
        if not inv:
            return "You are not carrying anything."
        return ", ".join(f"{amt} {item_id_to_str(item)}" for item, amt in inv.items())

    def _compute_closure_info(self):
        """Precompute the goal's recipe closure (ground-truth) so process-reward
        can judge whether a crafted/gathered item advances the goal."""
        tree = self._env.crafting_tree
        goal = self._env.goal
        try:
            closure_recipes = tree.traverse_recipe_tree(goal, set())
        except Exception:
            closure_recipes = []
        closure_items = set()
        closure_inputs = set()
        for recipe in closure_recipes:
            out = recipe.output_item.item_tag
            closure_items.add(out.item_id)
            closure_items.add(out.tag)
            for inp in recipe.input_items:
                closure_inputs.add(inp.item_tag.item_id)
                closure_inputs.add(inp.item_tag.tag)
        closure_items.add(goal)
        closure_items.discard(None)
        closure_inputs.discard(None)
        self._closure_items = closure_items
        self._closure_inputs = closure_inputs

    def _get_g_set(self) -> set:
        """Ground-truth progress state: the set of closure nodes already
        completed (intermediates crafted) or materials held, per current inventory.
        Used as (a) the structured MICo anchor d_g and (b) the g_text source."""
        inv = self._env.inventory
        g = set()
        for node in self._closure_items:
            if node in inv and inv[node] > 0:
                g.add(node)
        for mat in self._closure_inputs:
            if mat in inv and inv[mat] > 0:
                g.add(mat)
        return g

    def _estimate_remaining_depth(self, g_set: set) -> int:
        """Greedy estimate of remaining crafting depth to the goal given the
        current progress set. Returns 0 when the goal is already craftable."""
        from .textcraft_src.utils import item_id_to_str
        if self._env.goal in g_set:
            return 0
        tree = self._env.crafting_tree
        goal = self._env.goal
        # BFS from goal down the recipe tree, counting levels until all
        # required inputs are in g_set (or are raw materials not in closure).
        def depth_of(item, seen):
            if item in g_set:
                return 0
            if item not in tree.itemid_recipes or item in seen:
                return 1  # raw material or cycle: one 'get' away
            seen = seen | {item}
            try:
                recipe = tree.itemid_recipes[item][0]
                return 1 + max(depth_of(i.item_tag.item_id, seen) for i in recipe.input_items)
            except (ValueError, IndexError, AttributeError):
                return 1
        try:
            return depth_of(goal, frozenset())
        except RecursionError:
            return -1

    def _build_progress_text(self) -> str:
        """Serialize the task-progress state g_t as text, to be prepended to
        the anchor observation. Captures WHICH parts are done/missing (not just
        a scalar percentage), directly addressing the combinatorial-state
        ambiguity of same-observation-different-progress rollouts."""
        from .textcraft_src.utils import item_id_to_str

        g_set = self._get_g_set()
        goal = self._env.goal

        completed = sorted(
            item_id_to_str(n) for n in self._closure_items
            if n != goal and n in g_set
        )
        missing = sorted(
            item_id_to_str(n) for n in self._closure_items
            if n != goal and n not in g_set
        )
        materials = sorted(
            f"{item_id_to_str(m)}: {self._env.inventory.get(m, 0)}"
            for m in self._closure_inputs
        )
        remaining_depth = self._estimate_remaining_depth(g_set)

        return (
            f"[Task Progress]\n"
            f"Completed subgoals: {', '.join(completed) if completed else 'none'}\n"
            f"Missing subgoals: {', '.join(missing) if missing else 'none'}\n"
            f"Materials in hand: {'; '.join(materials) if materials else 'none'}\n"
            f"Remaining crafting depth: {remaining_depth}"
        )

    def _compute_process_reward(self, action: str, gained: dict, won: bool) -> float:
        """Rule-based PRM: +1 on goal; otherwise judge whether this (obs -> action)
        step produced an item inside / outside the goal's recipe closure."""
        if won:
            return 1.0
        if action.startswith("craft"):
            if not gained:
                return -0.1  # craft failed (insufficient materials / invalid)
            for item_id in gained:
                return 0.5 if item_id in self._closure_items else -0.3
        elif action.startswith("get"):
            if not gained:
                return -0.1
            for item_id in gained:
                return 0.2 if item_id in self._closure_inputs else -0.1
        return 0.0  # inventory / anything else

    def reset(self, goal_item: str, depth: int):
        self._env.inventory = {}
        self._env.goal = goal_item
        self._step_count = 0
        self._goal_item = goal_item
        self._goal_depth = int(depth)
        self._compute_closure_info()

        obs = self._build_initial_observation()
        info = {
            "won": False,
            "task_score": 0.0,
            "goal_item": goal_item,
            "depth": int(depth),
            "step_count": 0,
            "inventory": self._inventory_str(),
        }
        return obs, info

    def step(self, action: str):
        before_inventory = dict(self._env.inventory)
        self._step_count += 1
        try:
            obs, raw_reward, terminated, _truncated, _info = self._env.step(action)
        except Exception as e:  # defensive: upstream throws bare exceptions sometimes
            obs = f"[ENV ERROR] {type(e).__name__}: {e}"
            raw_reward, terminated = 0, False

        truncated = self._step_count >= self._max_steps
        done = bool(terminated) or truncated
        won = bool(terminated and raw_reward == 1)
        reward = self._reward_scale if won else 0.0

        # Rule-based process reward (PRM): score this (obs -> action) step against
        # the known recipe closure of the goal. Ground-truth, no learned model.
        gained = {}
        for item_id, amt in self._env.inventory.items():
            prev = before_inventory.get(item_id, 0)
            if amt > prev:
                gained[item_id] = amt - prev
        process_reward = self._compute_process_reward(action, gained, won)

        info = {
            "won": won,
            "task_score": 1.0 if won else 0.0,
            "goal_item": self._goal_item,
            "depth": int(self._goal_depth),
            "step_count": self._step_count,
            "inventory": self._inventory_str(),
            "process_reward": process_reward,
            "g_set": self._get_g_set(),
            "g_text": self._build_progress_text(),
        }
        return obs, reward, done, info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# MultiProcessEnv wrapper -----------------------------------------------------
# -----------------------------------------------------------------------------


class TextCraftMultiProcessEnv(gym.Env):
    """Ray-backed vectorised wrapper around TextCraft.

    `info` dicts always contain `won`, `goal_item`, `depth`, `step_count`.
    """

    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train:
            assert group_n == 1, "val mode requires group_n == 1"

        self._rng = np.random.RandomState(seed)
        self._env_kwargs = dict(env_kwargs) if env_kwargs else {}
        self._env_kwargs.setdefault("max_steps", 30)

        env_worker = ray.remote(**resources_per_worker)(TextCraftWorker)
        self._workers = [env_worker.remote(seed + (i // self.group_n), self._env_kwargs) for i in range(self.num_processes)]

        # depth annotations + split list
        self._goal_index: Dict[str, int] = _load_goal_index()
        pool = _load_split(is_train)
        # Drop any pool entries we don't have depth annotations for (shouldn't
        # happen, but defensive).
        self._sample_pool = [g for g in pool if g in self._goal_index]
        if not self._sample_pool:
            raise RuntimeError("TextCraft sample pool is empty after intersecting split with goal index — regenerate textcraft_splits.json.")
        print(
            f"[TextCraftEnvs] split={'train' if is_train else 'val'} goal_pool={len(self._sample_pool)} depth_dist={self._depth_distribution()}",
            flush=True,
        )

        # Val: deterministic shuffled cursor so successive val cycles cover
        # different goals; train: seeded random.choice each reset.
        self._val_cursor = 0
        if not is_train:
            perm = self._rng.permutation(len(self._sample_pool))
            self._val_perm = [self._sample_pool[i] for i in perm]

    def _depth_distribution(self) -> Dict[int, int]:
        from collections import Counter

        return dict(Counter(self._goal_index[g] for g in self._sample_pool))

    # -------------------------------------------------------------------------

    def step(self, actions):
        if len(actions) != self.num_processes:
            raise ValueError(f"Expected {self.num_processes} actions, got {len(actions)}")
        futures = [w.step.remote(a) for w, a in zip(self._workers, actions)]
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        if self.is_train:
            size = min(self.env_num, len(self._sample_pool))
            replace = size < self.env_num
            idx = self._rng.choice(len(self._sample_pool), size=self.env_num, replace=replace)
            picks = [self._sample_pool[int(i)] for i in idx]
        else:
            n = len(self._val_perm)
            picks = [self._val_perm[(self._val_cursor + k) % n] for k in range(self.env_num)]
            self._val_cursor = (self._val_cursor + self.env_num) % n
        # Replicate each pick across the group so a GRPO group sees the same goal.
        picks = [picks[i // self.group_n] for i in range(self.num_processes)]

        futures = [w.reset.remote(goal, self._goal_index[goal]) for w, goal in zip(self._workers, picks)]
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    # -------------------------------------------------------------------------

    def close(self):
        if getattr(self, "_closed", False):
            return
        try:
            ray.get([w.close.remote() for w in self._workers])
        except Exception:
            pass
        for w in self._workers:
            try:
                ray.kill(w)
            except Exception:
                pass
        self._closed = True

    def __del__(self):
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------


def build_textcraft_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    return TextCraftMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )

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

import ray
import gym
import numpy as np

# -----------------------------------------------------------------------------
# Progress tracking (mirrors AlfProgressTracker; see DESIGN_PROGRESS_AWARE_BISIM_HEAD.md)
# -----------------------------------------------------------------------------

class WebshopProgressTracker:
    """Per-episode cumulative shopping-progress tracker.

    g_set elements are (predicate, object) tuples:
        ("search", n)   — the n-th distinct search issued (query text is NOT part
                          of the key: two states reached by different queries for
                          the same goal should stay bisimilar)
        ("view", asin)  — product detail page visited (browsing depth = progress)
        ("buy", asin)   — a purchase (terminal)
    Pagination / back / description / features / reviews clicks are navigation and
    deliberately excluded, so states differing only in browsing position remain
    bisimilar (same rationale as ALFWorld navigation exclusion).

    The tracker consumes the *executed action* + env status rather than parsing
    observation text: WebshopWorker.step already knows the action string and
    whether the step terminated (buy) — this is more robust than regexing obs.
    """

    # Clickables that are pure navigation (excluded from g_set)
    _NAV_CLICKABLES = {
        "back to search",
        "< prev",
        "next >",
        "next",
        "prev",
        "buy now",  # handled separately as ("buy", asin)
    }
    _ITEM_SUB_PAGES = {"description", "features", "reviews"}

    def __init__(self):
        self.n_searches = 0
        self.viewed_items = []  # ordered unique asins
        self.purchased = None  # asin or None

    def update(self, action: str, done: bool, page_source_is_item: bool) -> None:
        """Accumulate one executed action.

        `page_source_is_item`: whether the *resulting* page is a product item
        page (used to count a view when the click was a product link).
        """
        a = action.strip().lower()
        if a.startswith("search[") and a.endswith("]"):
            self.n_searches += 1
            return
        if not a.startswith("click[") or not a.endswith("]"):
            return
        target = a[len("click[") : -1].strip()
        if target == "buy now":
            # asin unknown at this layer; caller supplies it via mark_purchase
            return
        if target in self._NAV_CLICKABLES or target in self._ITEM_SUB_PAGES:
            return
        if page_source_is_item and target not in self.viewed_items:
            self.viewed_items.append(target)
        return

    def mark_purchase(self, won: bool) -> None:
        self.purchased = "correct" if won else "wrong"

    def g_set(self):
        s = {("search", i + 1) for i in range(self.n_searches)}
        s |= {("view", asin) for asin in self.viewed_items}
        if self.purchased is not None:
            s.add(("buy", self.purchased))
        return frozenset(s)

    def g_text(self) -> str:
        if self.purchased is not None:
            return f"[Progress] purchase completed ({self.purchased})"
        parts = []
        if self.n_searches:
            parts.append(f"searches issued: {self.n_searches}")
        if self.viewed_items:
            items = ", ".join(self.viewed_items[:8])
            more = " ..." if len(self.viewed_items) > 8 else ""
            parts.append(f"items viewed: {items}{more}")
        if not parts:
            return ""
        return f"[Progress] {'; '.join(parts)}"


def _webshop_g_info(tracker: WebshopProgressTracker, won) -> dict:
    info = {"g_set": tracker.g_set(), "g_text": tracker.g_text()}
    if won is not None:
        info["g_progress_frac"] = 1.0 if won else None
    return info


# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray actor hosting *N logical envs* that share ONE SimServer.

    Memory note (OOM fix, 2026-09-13): the full WebShop product catalog
    (items_shuffle.json, 1.18M items) costs ~17.7GB resident per process.
    With one actor per logical env (256 per run: 16*8 train + 128 val) that
    is ~4.5TB — far beyond the node's 1.9TB and Ray killed the run
    (OutOfMemoryError). Here a single actor loads the catalog once and hosts
    `n_envs` WebAgentTextEnv instances that share that SimServer; sessions
    are isolated via per-env `session_prefix` (SimServer keeps all state in
    `user_sessions[session_id]`). WebShop steps are cheap (template render +
    Lucene lookup), so serving 32 logical envs serially per actor is fine.

    API: reset(env_idx, goal_idx) / step(env_idx, action) route to the i-th
    logical env; the per-env WebshopProgressTracker is kept in `self._trackers`.
    """

    def __init__(self, seed, env_kwargs, n_envs=32, server_seed=None):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv  # noqa: WPS433

        env_kwargs = dict(env_kwargs)
        base_url = 'http://127.0.0.1:3000'
        # ONE shared server per actor: catalog + Lucene index loaded once.
        # server_seed (CRITICAL, 2026-09-15 fix): SimServer shuffles its goals
        # list with this seed. The manager broadcasts ONE goal_idx per rollout
        # group, and every actor maps that idx through its OWN goals list — so
        # if actors use different seeds, a group whose envs straddle an actor
        # boundary (envs_per_actor not divisible by group_n) would mix DIFFERENT
        # goals inside one GRPO group. All actors therefore share one canonical
        # server_seed (= the WebshopMultiProcessEnv base seed), making the
        # goals permutation identical everywhere. Per-env kwargs seeds below
        # only feed WebAgentTextEnv's own RNG (unused: '<r>' random search is
        # unreachable for trained policies), so cross-actor collisions there
        # are harmless.
        srv_seed = server_seed if server_seed is not None else seed
        self._server = SimServer(
            srv_seed,
            base_url,
            env_kwargs.get('file_path'),
            env_kwargs.get('attr_path'),
            None,                       # filter_goals
            -1,                         # limit_goals
            env_kwargs.get('num_products'),
            env_kwargs.get('human_goals', 1),
            False,                      # show_attrs
        )
        self._envs, self._trackers = [], []
        for i in range(n_envs):
            kwargs = dict(env_kwargs)
            kwargs['seed'] = seed + i
            kwargs['session_prefix'] = f'e{i}_'
            # Direct construction (no gym.make) — server injection + no wrapper noise.
            env = WebAgentTextEnv(
                observation_mode=kwargs.get('observation_mode', 'text'),
                file_path=kwargs.get('file_path'),
                attr_path=kwargs.get('attr_path'),
                server=self._server,
                **{k: v for k, v in kwargs.items()
                   if k not in ('observation_mode', 'file_path', 'attr_path')},
            )
            self._envs.append(env)
            self._trackers.append(WebshopProgressTracker())

    # ------------------------------------------------------------------
    # Per-logical-env API (env_idx in [0, n_envs))
    # ------------------------------------------------------------------

    @staticmethod
    def _page_is_item_page(env) -> bool:
        """Whether the browser currently shows a product item page."""
        try:
            from web_agent_site.engine.engine import get_page_name  # type: ignore
            return get_page_name(env.browser.current_url) == "item_page"
        except Exception:
            return "item_page" in (env.browser.current_url or "")

    def step(self, env_idx: int, action: str):
        """Execute a step in the i-th logical environment"""
        env, progress = self._envs[env_idx], self._trackers[env_idx]
        obs, reward, done, info = env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = env.get_available_actions()
        info['task_score'] = reward

        won = bool(done and reward == 1.0)

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        # --- progress tracking (executed-action based, not obs parsing) ---
        g_before = len(progress.g_set())
        n_views_before = len(progress.viewed_items)
        progress.update(action, done, self._page_is_item_page(env))
        if done:
            progress.mark_purchase(won)
        # Rule-based PRM (TextCraft/ALFWorld-compatible semantics):
        #   +1.0 correct purchase (win), -0.5 wrong purchase,
        #   +0.2 NEW item page viewed, -0.1 malformed action, 0 search/navigation.
        # Searches change g_set too, but they are not browsing progress — only
        # count the *view* events for the +0.2 tier.
        if done:
            process_reward = 1.0 if won else -0.5
        else:
            a = action.strip().lower()
            if a.startswith(("search[", "click[")):
                process_reward = 0.2 if len(progress.viewed_items) > n_views_before else 0.0
            else:
                process_reward = -0.1
        info['process_reward'] = process_reward
        info.update(_webshop_g_info(progress, won if done else None))

        return obs, reward, done, info

    def reset(self, env_idx: int, goal_idx: int):
        """Reset the i-th logical environment with the given goal/session index"""
        env = self._envs[env_idx]
        obs, info = env.reset(session=goal_idx)
        info = dict(info or {})
        info['available_actions'] = env.get_available_actions()
        info['won'] = False
        self._trackers[env_idx] = WebshopProgressTracker()
        info['process_reward'] = 0.0
        info.update(_webshop_g_info(self._trackers[env_idx], None))
        return obs, info

    def render(self, env_idx: int, mode_for_render: str):
        return self._envs[env_idx].render(mode=mode_for_render)

    def get_available_actions(self, env_idx: int):
        return self._envs[env_idx].get_available_actions()

    def get_goals(self):
        return self._server.goals

    def close(self):
        for env in self._envs:
            try:
                env.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        envs_per_actor: int = 32,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        # OOM fix (2026-09-13): shard logical envs onto FEW actors, each of
        # which loads the 1.18M-product catalog ONCE (~17.7GB) and hosts
        # `envs_per_actor` WebAgentTextEnv instances sharing that SimServer.
        # (One-actor-per-env = 256 processes × 17.7GB ≈ 4.5TB → Ray OOM-kill.)
        # resources_per_worker now applies PER ACTOR: default num_cpus=0.5 with
        # 32 envs/actor keeps the CPU budget identical to the old scheme.
        n_actors = (self.num_processes + envs_per_actor - 1) // envs_per_actor
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        # env i -> (actor index, local env index within that actor)
        self._env_owner = []
        for a in range(n_actors):
            n_local = min(envs_per_actor, self.num_processes - a * envs_per_actor)
            # server_seed=seed (canonical): every actor gets the SAME goals
            # permutation so a goal_idx broadcast to a group maps to the same
            # goal regardless of which actor serves each member env (see the
            # WebshopWorker.__init__ comment for the cross-actor bug this fixes).
            # Per-actor seed (seed+a) still feeds the per-env WebAgentTextEnv RNGs.
            self._workers.append(env_worker.remote(seed + a, self._env_kwargs, n_local, server_seed=seed))
            for l in range(n_local):
                self._env_owner.append((a, l))

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Route each logical env's action to its owning actor
        futures = []
        for i, action in enumerate(actions):
            a, l = self._env_owner[i]
            futures.append(self._workers[a].step.remote(l, action))

        # Collect results (keep the original logical-env order)
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Route each logical env's reset to its owning actor
        futures = []
        for i, goal_idx in enumerate(idx):
            a, l = self._env_owner[i]
            futures.append(self._workers[a].reset.remote(l, goal_idx))

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            a, l = self._env_owner[env_idx]
            return ray.get(self._workers[a].render.remote(l, mode))

        futures = []
        for i in range(self.num_processes):
            a, l = self._env_owner[i]
            futures.append(self._workers[a].render.remote(l, mode))
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
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

import re
from typing import List

# Safe no-op shipped to TextCraft when the model output cannot be parsed. `inventory`
# is the right choice: it is a valid action (regex `inventory` in env.py), it does not
# mutate state, and it produces a deterministic short observation. We must not send
# the raw model output because TextCraft will treat anything unparseable as a failed
# action and emit a verbose error message, which pollutes context length.
_INVALID_ACTION_NOOP = "inventory"


def textcraft_projection(actions: List[str]):
    """Extract the body of <action>...</action> from each raw model output and
    mark validity. Also requires a matching <think>...</think> block and rejects
    Chinese characters.

    Returns (actions, valids). On parse failure, the action string is replaced by
    a no-op `inventory` action so the env advances cleanly without corrupting state.
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]
        lower = actions[i].lower()

        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = lower.find(start_tag)
        end_idx = lower.find(end_tag)

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            actions[i] = _INVALID_ACTION_NOOP
            continue

        try:
            extracted = lower[start_idx + len(start_tag) : end_idx].strip()
            if not extracted:
                actions[i] = _INVALID_ACTION_NOOP
                continue
            actions[i] = extracted
            valids[i] = 1
        except Exception:
            actions[i] = _INVALID_ACTION_NOOP
            continue

        # Format reward: also require a <think>...</think> block in the raw output.
        if original_str.find("<think>") == -1 or original_str.find("</think>") == -1:
            valids[i] = 0

        # Chinese characters in the output: reject.
        if re.search(r"[一-鿿]", original_str):
            valids[i] = 0

    return actions, valids

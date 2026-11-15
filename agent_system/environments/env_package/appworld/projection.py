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

from typing import List

# Safe no-op shipped to AppWorld when the model output cannot be parsed. We must not send
# arbitrary trailing text because AppWorld will execute it as Python and either crash the
# step or, worse, corrupt server state. A bare print keeps the trajectory advancing while
# letting the format-reward signal (valids[i] == 0) do its job.
_INVALID_ACTION_NOOP = "print('[INVALID FORMAT: missing <code>...</code> block]')"


def appworld_projection(actions: List[str]):
    """
    Process raw model outputs into AppWorld-executable code strings.

    Returns the in-place patched action list and a parallel `valids` list where 1 means the
    format was well-formed (both <think>...</think> and <code>...</code> present).
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]

        start_tag = "<code>"
        end_tag = "</code>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            actions[i] = _INVALID_ACTION_NOOP
            valids[i] = 0
            continue

        actions[i] = actions[i][start_idx + len(start_tag) : end_idx]
        valids[i] = 1

        # Format reward: require a <think>...</think> block too. We don't overwrite the
        # extracted code, just downgrade the format flag.
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1 or think_end_idx <= think_start_idx:
            valids[i] = 0

    return actions, valids

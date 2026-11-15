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
import re


def _strip_action_prefix(action: str) -> str:
    action = action.strip()
    if action.lower().startswith("action:"):
        action = action.split(":", 1)[1].strip()
    return action


def sciworld_projection(actions: List[str]):
    """
    Project model text into ScienceWorld text actions.

    Expected format:
        <think>reasoning...</think><action>look around</action>
    """
    processed_actions = []
    valids = [0] * len(actions)

    for i, action_text in enumerate(actions):
        original_str = action_text
        lower_str = action_text.lower()

        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = lower_str.find(start_tag)
        end_idx = lower_str.find(end_tag)

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            processed_actions.append(original_str[-20:].strip())
            continue

        extracted_action = original_str[start_idx + len(start_tag):end_idx]
        processed_actions.append(_strip_action_prefix(extracted_action))
        valids[i] = 1

        think_start_idx = lower_str.find("<think>")
        think_end_idx = lower_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1 or think_end_idx <= think_start_idx:
            valids[i] = 0

        if re.search(r"[\u4e00-\u9fff]", original_str):
            valids[i] = 0

    return processed_actions, valids

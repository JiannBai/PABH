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

SCIWORLD_ACTIONS = """
[
{"action": "open OBJ", "description": "open a container or door"},
{"action": "close OBJ", "description": "close a container or door"},
{"action": "activate OBJ", "description": "activate a device"},
{"action": "deactivate OBJ", "description": "deactivate a device"},
{"action": "connect OBJ to OBJ", "description": "connect electrical components"},
{"action": "disconnect OBJ", "description": "disconnect electrical components"},
{"action": "use OBJ [on OBJ]", "description": "use a device or item"},
{"action": "look around", "description": "describe the current room"},
{"action": "look at OBJ", "description": "describe an object in detail"},
{"action": "look in OBJ", "description": "describe a container's contents"},
{"action": "read OBJ", "description": "read a note or book"},
{"action": "move OBJ to OBJ", "description": "move an object to a container"},
{"action": "pick up OBJ", "description": "move an object to the inventory"},
{"action": "put down OBJ", "description": "drop an inventory item"},
{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"},
{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"},
{"action": "mix OBJ", "description": "chemically mix a container"},
{"action": "go to LOC", "description": "move to a new location"},
{"action": "eat OBJ", "description": "eat a food"},
{"action": "flush OBJ", "description": "flush a toilet"},
{"action": "focus on OBJ", "description": "signal intent on a task object"},
{"action": "wait", "description": "take no action for 10 iterations"},
{"action": "wait1", "description": "take no action for 1 iteration"},
{"action": "task", "description": "describe current task"},
{"action": "inventory", "description": "list your inventory"}
]
""".strip()

SCIWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ScienceWorld environment, a text-based science task simulator.
Your current task is: {task_description}

Your current observation is: {current_observation}

General action templates:
{general_actions}

Current action hints from the environment:
{available_actions}

Now choose one action for the current step.
First reason step by step inside <think> </think> tags.
Then output exactly one environment action inside <action> </action> tags.
""".strip()

SCIWORLD_TEMPLATE = """
You are an expert agent operating in the ScienceWorld environment, a text-based science task simulator.
Your current task is: {task_description}

Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and corresponding actions:
{action_history}

You are now at step {current_step}.
Your current observation is: {current_observation}

General action templates:
{general_actions}

Current action hints from the environment:
{available_actions}

Now choose one action for the current step.
First reason step by step inside <think> </think> tags.
Then output exactly one environment action inside <action> </action> tags.
""".strip()


SCIWORLD_TEMPLATE_NO_HIS_TEMPLATE_ONLY = """
You are an expert agent operating in the ScienceWorld environment, a text-based science task simulator.
Your current task is: {task_description}

Your current observation is: {current_observation}

General action templates:
{general_actions}

Replace OBJ/LOC placeholders with exact object or location names mentioned in the current observation.

Now choose one action for the current step.
First reason step by step inside <think> </think> tags.
Then output exactly one environment action inside <action> </action> tags.
""".strip()

SCIWORLD_TEMPLATE_TEMPLATE_ONLY = """
You are an expert agent operating in the ScienceWorld environment, a text-based science task simulator.
Your current task is: {task_description}

Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and corresponding actions:
{action_history}

You are now at step {current_step}.
Your current observation is: {current_observation}

General action templates:
{general_actions}

Replace OBJ/LOC placeholders with exact object or location names mentioned in the current observation or recent history.

Now choose one action for the current step.
First reason step by step inside <think> </think> tags.
Then output exactly one environment action inside <action> </action> tags.
""".strip()

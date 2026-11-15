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

# --------------------- TextCraft --------------------- #
# TextCraft (ADaPT, EMNLP'24) is a Minecraft-recipe crafting environment.
# Three valid action types, free-form within the <action> tag:
#   craft <N> <item> using <N1> <ing1>, <N2> <ing2>, ...
#   get <N> <item>
#   inventory

TEXTCRAFT_TEMPLATE_NO_HIS = """
You are an expert agent operating in the TextCraft interactive crafting environment, modeled after Minecraft. Your goal is to craft a target item by combining ingredients along a chain of recipes.

Your current observation is:
{current_observation}

Current inventory: {inventory}

You can issue exactly one action per step, using one of these three formats:
  - `craft <out_count> <out_item> using <in_count_1> <in_item_1>, <in_count_2> <in_item_2>, ...` — apply a recipe (the inputs must already be in your inventory).
  - `get <count> <item>` — fetch a raw (non-craftable) base material directly from the environment.
  - `inventory` — print what you currently hold.

Items are spelled out in lower-case with spaces between words (e.g. `oak planks`, not `minecraft:oak_planks`).

Recipe matching rules (important — follow all of these):
  - Use ONLY the crafting commands shown in the recipe list; do not invent new recipes.
  - The target item AND its count must match a recipe line EXACTLY. E.g. if the recipe says `craft 3 dark oak sign using ...`, write `craft 3 dark oak sign ...`, not `craft 1 dark oak sign ...`. You cannot use a partial recipe.
  - The ingredient counts must ALSO match the recipe EXACTLY. E.g. if the recipe says `using 1 oak logs`, write `using 1 oak logs`, not `using 2 oak logs`.
  - The target item must be a SPECIFIC item (e.g. `dark oak sign`, `oak planks`). A generic family name as the target (e.g. just `planks` or `logs`) is not a valid item and will fail.
  - If a recipe lists a GENERIC ingredient (e.g. `using 4 planks`), you MUST substitute a specific variant from the same family (e.g. `using 4 oak planks`) in your craft command — generic ingredient names are tags and the env cannot match them against your inventory.
  - It is fine to produce more of the target than your goal requires; if a single craft does not yield enough, repeat the craft command in a later step (each craft consumes another full set of ingredients).

Example of one well-formed turn (for format only; the recipes and goal differ from yours):
  Recipes available include: `craft 3 dark oak sign using 6 dark oak planks, 1 stick`, `craft 1 fletching table using 4 planks, 2 flint`.
  Goal: craft fletching table. Current inventory: 4 oak planks, 2 flint.
  <think>I need a fletching table. The recipe needs 4 planks (generic) and 2 flint. I have 4 oak planks and 2 flint, so I substitute `oak planks` for the generic `planks` and craft directly.</think>
  <action>craft 1 fletching table using 4 oak planks, 2 flint</action>

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, what intermediate items you already have, and which next recipe step brings you closest to the goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an action for current step and present it within <action> </action> tags. Output exactly one action; do not chain multiple actions in one response.
"""

TEXTCRAFT_TEMPLATE = """
You are an expert agent operating in the TextCraft interactive crafting environment, modeled after Minecraft. Your goal is to craft a target item by combining ingredients along a chain of recipes.

Task context (constant across the episode):
{task_context}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took:
{action_history}

You are now at step {current_step} and your current observation is:
{current_observation}

Current inventory: {inventory}

You can issue exactly one action per step, using one of these three formats:
  - `craft <out_count> <out_item> using <in_count_1> <in_item_1>, <in_count_2> <in_item_2>, ...` — apply a recipe (the inputs must already be in your inventory).
  - `get <count> <item>` — fetch a raw (non-craftable) base material directly from the environment.
  - `inventory` — print what you currently hold.

Items are spelled out in lower-case with spaces between words (e.g. `oak planks`, not `minecraft:oak_planks`).

Recipe matching rules (important — follow all of these):
  - Use ONLY the crafting commands shown in the recipe list; do not invent new recipes.
  - The target item AND its count must match a recipe line EXACTLY. E.g. if the recipe says `craft 3 dark oak sign using ...`, write `craft 3 dark oak sign ...`, not `craft 1 dark oak sign ...`. You cannot use a partial recipe.
  - The ingredient counts must ALSO match the recipe EXACTLY. E.g. if the recipe says `using 1 oak logs`, write `using 1 oak logs`, not `using 2 oak logs`.
  - The target item must be a SPECIFIC item (e.g. `dark oak sign`, `oak planks`). A generic family name as the target (e.g. just `planks` or `logs`) is not a valid item and will fail.
  - If a recipe lists a GENERIC ingredient (e.g. `using 4 planks`), you MUST substitute a specific variant from the same family (e.g. `using 4 oak planks`) in your craft command — generic ingredient names are tags and the env cannot match them against your inventory.
  - It is fine to produce more of the target than your goal requires; if a single craft does not yield enough, repeat the craft command in a later step (each craft consumes another full set of ingredients).

Example of one well-formed turn (for format only; the recipes and goal differ from yours):
  Recipes available include: `craft 3 dark oak sign using 6 dark oak planks, 1 stick`, `craft 1 fletching table using 4 planks, 2 flint`.
  Goal: craft fletching table. Current inventory: 4 oak planks, 2 flint.
  <think>I need a fletching table. The recipe needs 4 planks (generic) and 2 flint. I have 4 oak planks and 2 flint, so I substitute `oak planks` for the generic `planks` and craft directly.</think>
  <action>craft 1 fletching table using 4 oak planks, 2 flint</action>

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, what intermediate items you already have, and which next recipe step brings you closest to the goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an action for current step and present it within <action> </action> tags. Output exactly one action; do not chain multiple actions in one response.
"""

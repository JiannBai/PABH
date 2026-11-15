#!/usr/bin/env python3
"""ScienceWorld g_text 表示修正的静态/反例测试。

对照 GTEXT_REPRESENTATION_FUTURE_WORK.md 的验证步骤 1/2：
  - 多参数动作保留第二参数（目的地/容器/工具/读数）；
  - 覆盖主动与被动反馈语态；
  - 房间描述/导航/无事件反馈不误记；
  - 有序去重、确定性序列化。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_system.environments.env_package.sciworld.sciworld_env_manager import (
    SciProgressTracker,
)


def parse(obs: str) -> str:
    t = SciProgressTracker()
    t.update(obs)
    return t.g_text()


def test_move_keeps_destination():
    a = parse("You move the thermometer to the inventory.")
    b = parse("You move the metal pot to the stove.")
    assert "move(thermometer→inventory)" in a, a
    assert "move(metal pot→stove)" in b, b
    assert a != b


def test_passive_state_changes():
    assert "activate(stove)" in parse("The stove is now activated.")
    assert "open(door)" in parse("The door is already open.")
    assert "deactivate(stove)" in parse("The stove is now deactivated.")


def test_heat_cool_covered():
    assert "heat(tin→stove)" in parse("You heat the tin using the stove.")
    assert "cool(water→freezer)" in parse("You cool the water using the freezer.")


def test_measure_covered():
    assert "measure(thermometer→40)" in parse(
        "the thermometer measures a temperature of 40 degrees celsius"
    )


def test_navigation_and_room_not_recorded():
    assert parse("You move to the kitchen.") == ""
    # 房间描述里 "is closed"（无 now/already）不应被记成 close 事件
    assert parse("This room is called the kitchen. The cupboard door is closed.") == ""


def test_dedup_and_normalization():
    t = SciProgressTracker()
    t.update("You move the thermometer to the inventory.")
    t.update("You move the thermometer to the inventory.")
    t.update("You move the metal pot to the stove.")
    out = t.g_text()
    assert out.count("move(thermometer→inventory)") == 1, out
    assert "move(metal pot→stove)" in out


def main():
    test_move_keeps_destination()
    test_passive_state_changes()
    test_heat_cool_covered()
    test_measure_covered()
    test_navigation_and_room_not_recorded()
    test_dedup_and_normalization()
    print("ALL ScienceWorld g_text checks PASSED")


if __name__ == "__main__":
    main()

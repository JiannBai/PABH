# Copyright 2026 BiPACE contributors.
"""Keep the reference BiPACE training budgets consistent across scripts."""

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BIPACE_EXAMPLES = REPO_ROOT / "examples/bipace"

EXPECTED_STEPS = {
    ("alfworld", "qwen1.5b"): 200,
    ("alfworld", "qwen7b"): 150,
    ("webshop", "qwen1.5b"): 200,
    ("webshop", "qwen7b"): 150,
    ("textcraft", "qwen1.5b"): 100,
    ("textcraft", "qwen7b"): 50,
}


def _resolve_shell_value(source: str, value: str) -> int:
    value = value.strip().strip('"').strip("'")
    if value.isdigit():
        return int(value)

    variable_match = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", value)
    assert variable_match, f"Unsupported shell value: {value}"
    variable = variable_match.group(1)
    assignment = re.search(rf"(?m)^{re.escape(variable)}=(\d+)\b", source)
    assert assignment, f"Cannot resolve ${variable}"
    return int(assignment.group(1))


@pytest.mark.parametrize(("environment", "model_dir", "expected"), [
    (*key, expected) for key, expected in EXPECTED_STEPS.items()
])
def test_bipace_training_horizons(environment: str, model_dir: str, expected: int):
    scripts = sorted((BIPACE_EXAMPLES / environment).glob(f"{model_dir}*/**/*.sh"))
    training_scripts = []

    for script in scripts:
        source = script.read_text(encoding="utf-8")
        if "python3 -m verl.trainer.main_ppo" not in source:
            continue
        training_scripts.append(script)

        epoch_values = re.findall(r"trainer\.total_epochs=([^\s\\]+)", source)
        assert epoch_values, f"Missing trainer.total_epochs in {script}"
        resolved_epochs = {_resolve_shell_value(source, value) for value in epoch_values}
        assert resolved_epochs == {expected}, f"{script}: expected {expected}, got {resolved_epochs}"

        if environment == "textcraft":
            cap_values = re.findall(r"trainer\.total_training_steps=([^\s\\]+)", source)
            assert cap_values, f"Missing hard training-step cap in {script}"
            resolved_caps = {_resolve_shell_value(source, value) for value in cap_values}
            assert resolved_caps == {expected}, f"{script}: expected hard cap {expected}, got {resolved_caps}"

    assert training_scripts, f"No training scripts found for {environment}/{model_dir}"

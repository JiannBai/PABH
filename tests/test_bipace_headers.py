"""Guard against copied upstream ownership headers in BiPACE-owned files."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_HEADER_TEXT = (
    "Nanyang "
    "Technological University",
    "verl-agent (GiGPO) "
    "team",
)

BIPACE_OWNED_FILES = [
    *sorted((REPO_ROOT / "gigpo").glob("*.py")),
    REPO_ROOT / "agent_system/__init__.py",
    REPO_ROOT / "agent_system/environments/env_package/__init__.py",
    *sorted((REPO_ROOT / "agent_system/environments/env_package/textcraft").glob("*.py")),
    REPO_ROOT / "agent_system/environments/prompts/textcraft.py",
    REPO_ROOT / "examples/data_preprocess/prepare_textcraft.py",
    REPO_ROOT / "examples/data_preprocess/prepare_eto_sft_webshop.py",
    REPO_ROOT / "examples/sft/webshop/train_sft_hf.py",
]

# These files are derived from verl-agent and must retain their upstream notice,
# while making BiPACE's modification ownership explicit on the first line.
UPSTREAM_DERIVED_FILES = {
    REPO_ROOT / "gigpo/core_gigpo.py",
    REPO_ROOT / "agent_system/environments/env_manager.py",
    REPO_ROOT / "agent_system/environments/env_package/appworld/envs.py",
    REPO_ROOT / "agent_system/environments/env_package/appworld/projection.py",
    REPO_ROOT / "agent_system/environments/prompts/__init__.py",
}
BIPACE_OWNED_FILES.extend(sorted(UPSTREAM_DERIVED_FILES))


def test_bipace_owned_files_do_not_claim_upstream_ownership():
    for path in BIPACE_OWNED_FILES:
        if path in UPSTREAM_DERIVED_FILES:
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            assert "BiPACE modifications" in first_line, path
            continue

        header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:5])
        for forbidden in FORBIDDEN_HEADER_TEXT:
            assert forbidden not in header, f"Copied upstream header in {path}"

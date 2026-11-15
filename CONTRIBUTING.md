# Contributing to BiPACE

Thank you for helping improve BiPACE. The project is built on verl-agent and
veRL, so contributions must preserve upstream license notices and attribution.

## Development setup

Install the repository in editable mode with test dependencies:

```bash
pip install -e ".[test]"
```

Environment-specific dependencies are intentionally installed separately; see
the root [README](./README.md#install-supported-environments).

## Before opening a pull request

Run the CPU-safe BiPACE checks:

```bash
python -m pytest -q \
  tests/test_bigpo.py \
  tests/test_cacb.py \
  tests/test_cacb_action_tag.py \
  tests/test_cacb_q_style.py \
  tests/test_nstep.py \
  tests/test_bipace_headers.py \
  tests/test_bipace_training_horizons.py
```

Also run `bash -n` on any changed shell scripts. GPU, Ray, and environment
integration tests should be reported when the change touches those paths.

## Project conventions

- Put reference training recipes under `examples/bipace/`.
- Keep matched baselines comparable in model, data, rollout budget, and seed.
- Preserve the documented 1.5B/7B training budgets unless the change is an
  explicitly named ablation.
- Add focused tests for estimator or grouping changes.
- Do not remove verl-agent, GiGPO, or veRL copyright and NOTICE text from
  files derived from those projects.

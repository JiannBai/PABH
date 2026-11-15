# BiPACE Training Scripts

This folder is the clean reproduction surface for BiPACE development on top of
verl-agent. It contains only the three supported benchmarks: ALFWorld, WebShop,
and TextCraft. The naming is intentionally compact:

- `train_bipace.sh`: full BiPACE configuration for that environment/model.
- `train_gigpo.sh`: GiGPO baseline.
- `train_bigpo.sh`: BigPO baseline.
- `train_bigpo_cacb.sh`: BigPO with CACB but without the full final variant.
- `train_grpo.sh`: GRPO baseline when used for that environment.
- `ablation/`: sweeps and targeted variants, such as action tags, epsilon,
  embedding choices, n-step, or seed bundles.

Run scripts from the repository root. Most scripts are self-contained and keep
their original training arguments so historical runs remain reproducible.

## Main Grid

| Environment | Model / hardware | Main scripts |
| --- | --- | --- |
| ALFWorld | `qwen1.5b_2gpu` | `alfworld/qwen1.5b_2gpu/train_bipace.sh`, `train_gigpo.sh`, `train_bigpo.sh` |
| ALFWorld | `qwen1.5b_4xh100` | `alfworld/qwen1.5b_4xh100/train_bigpo_cacb.sh`, `train_gigpo.sh`, `train_bigpo.sh` |
| ALFWorld | `qwen7b_4xh100` | `alfworld/qwen7b_4xh100/train_bipace.sh`, `train_gigpo.sh`, `train_bigpo.sh` |
| TextCraft | `qwen1.5b_4xh100` | `textcraft/qwen1.5b_4xh100/train_bipace.sh`, `train_gigpo.sh`, `train_grpo.sh` |
| TextCraft | `qwen7b_4xh100` | `textcraft/qwen7b_4xh100/train_bipace.sh`, `train_gigpo.sh`, `train_grpo.sh` |
| WebShop | `qwen1.5b_2gpu` | `webshop/qwen1.5b_2gpu/train_bipace.sh`, `train_gigpo.sh`, `train_bigpo.sh` |
| WebShop | `qwen7b_4xh100` | `webshop/qwen7b_4xh100/train_bipace.sh`, `train_gigpo.sh`, `train_bigpo.sh` |

## Conventions

- Use `SEED=...` or script-specific seed lists when provided.
- Override model and data paths via the variables already exposed inside each
  script.
- Logs are generated under each script's configured `train_logs/` and
  `swanlog_*` paths; these outputs are intentionally not checked in.

## Training Budgets

These budgets apply to every training script under this directory, including
the corresponding ablations. Launcher-only and offline-analysis scripts do not
carry a training budget.

| Environment | Qwen2.5-1.5B | Qwen2.5-7B |
| --- | ---: | ---: |
| ALFWorld | 200 steps | 150 steps |
| WebShop | 200 steps | 150 steps |
| TextCraft | 100 steps | 50 steps |

TextCraft scripts additionally set `trainer.total_training_steps` as a hard
cap. ALFWorld and WebShop use one full-batch update per epoch, so their
`trainer.total_epochs` values encode the step budgets above.

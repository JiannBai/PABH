# Examples

This directory keeps upstream verl-agent examples and the BiPACE reproduction
entry points separate.

- `bipace/`: BiPACE reference training scripts for ALFWorld, WebShop, and
  TextCraft. Start here for main runs, baselines, and ablations.
- `baselines/`: extra baseline-only scripts that are not part of the main
  BiPACE run grid.
- `archive/`: older seed-specific, exploratory, or superseded scripts kept for
  provenance.
- `data_preprocess/`, `sft/`, `search/`, `env_server/`: data preparation,
  supervised fine-tuning, retriever, and service helpers.
- `*_trainer/`: upstream verl-agent trainer examples retained as references.

Unless a script says otherwise, run scripts from the repository root, for
example:

```bash
bash examples/bipace/webshop/qwen7b_4xh100/train_bipace.sh
```

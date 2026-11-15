# BiPACE pull request

### Checklist before starting

- [ ] Search for similar PR(s).
- [ ] Read [CONTRIBUTING.md](../CONTRIBUTING.md).

### What does this PR do?

> Add one-line overview of what this PR aims to achieve or accomplish. 

### High-Level Design

> Demonstrate the high-level design if this PR is complex.

### Specific Changes

> List the specific changes.

### API

> Demonstrate how the API changes if any.

### Usage Example

> Provide usage example(s) for easier usage.

```python
# Add code snippet or script demonstrating how to use this 
```

### Tests and evidence

> List CPU/GPU tests. For estimator, model, or environment changes that cannot
> be covered by public CI, include matched experimental evidence and the exact
> recipe, seed, model, and training budget.

### Additional information

- **Issue Number**: Fixes issue # or discussion # if any.
- **Training**: [Note which backend this PR will affect: FSDP, Megatron, both, or none]
- **Inference**: [Note which backend this PR will affect: vLLM, SGLang, both, or none]

### Checklist before submitting

- [ ] Run the CPU-safe checks in [CONTRIBUTING.md](../CONTRIBUTING.md).
- [ ] Run `bash -n` on changed shell scripts.
- [ ] Preserve verl-agent, GiGPO, and veRL attribution in derived files.
- [ ] Keep reference comparisons matched in model, data, rollout budget, and seed.
- [ ] Add `[BREAKING]` to the PR title if it breaks any API.
- [ ] Update BiPACE documentation or recipes when behavior changes.
- [ ] Add CI test(s) if necessary.

# Copyright 2026 BiPACE contributors.
"""BiPACE and GiGPO advantage-estimation components.

The package intentionally avoids eager imports so lightweight utilities and
configuration checks do not load PyTorch or optional embedding dependencies.
"""

__all__ = [
    "bigpo_logger",
    "core_bigpo",
    "core_cacb",
    "core_gigpo",
]

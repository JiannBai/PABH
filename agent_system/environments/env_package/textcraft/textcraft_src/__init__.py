# Vendored TextCraft source (MIT licensed).
# Upstream: https://github.com/archiki/ADaPT/tree/main/TextCraft
# Upstream packaging: https://pypi.org/project/textcraft/ (textcraft 0.0.3, MIT)
#
# We vendor instead of pip-installing so the env package is hermetic and so we can
# patch the upstream code (relative imports, optional surface of crafting depth in
# info dicts) without monkey-patching a third-party package.
from .crafting_tree import CraftingTree
from .env import TextCraft
from .utils import ActionFailed, ItemTag, ItemTagWithCount, Recipe, item_id_to_str

__all__ = [
    "ActionFailed",
    "CraftingTree",
    "ItemTag",
    "ItemTagWithCount",
    "Recipe",
    "TextCraft",
    "item_id_to_str",
]

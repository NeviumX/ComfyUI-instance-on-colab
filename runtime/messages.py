"""The texts the runtime shows, in the language of the notebook (translations.json)."""

import json
from pathlib import Path


MESSAGES = None  # build_notebook.py puts the messages of the notebook's language on this line.
if MESSAGES is None:
    # Run from the source tree: the Japanese messages of translations.json next to this package.
    MESSAGES = {f"{section}.{name}": entry["JP"] for section, entries in json.loads(
        Path(__file__).resolve().parent.parent.joinpath("translations.json").read_text(encoding="utf-8"))["runtime"].items()
        for name, entry in entries.items()}


def t(key, **values):
    """The message translations.json holds as runtime.<key>, with its values filled in."""
    return MESSAGES[key].format(**values)

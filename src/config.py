"""
Access to configs/config.yaml.

The file is parsed once and cached — scoring runs read from it per request, so
re-reading from disk each time would put a YAML parse in the hot path.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Return the parsed configuration. Cached after the first call."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_section(name: str) -> dict[str, Any]:
    """
    Return a top-level configuration section.

    Args:
        name: Section key, e.g. 'scoring' or 'evaluation'.

    Raises:
        KeyError: If the section is missing from config.yaml.
    """
    config = load_config()
    if name not in config:
        raise KeyError(f"Missing '{name}' section in {CONFIG_PATH}")
    return config[name]


def reload_config() -> None:
    """Drop the cached configuration. Intended for tests."""
    load_config.cache_clear()

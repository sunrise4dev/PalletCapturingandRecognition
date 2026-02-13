"""Configuration loading and validation utilities."""

import os
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load a YAML config file and resolve relative paths."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge two config dicts (override takes precedence)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def get_project_root() -> Path:
    """Get the project root directory (where setup.py lives)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists():
            return parent
    return Path.cwd()

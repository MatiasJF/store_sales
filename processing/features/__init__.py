"""
Feature registry: auto-discovers and manages all feature functions.

Each feature function is decorated with @register and takes/returns a DataFrame.
The optimizer can toggle feature groups on/off during search.
"""

_REGISTRY: dict[str, dict] = {}


def register(name: str, group: str):
    """Decorator to register a feature function."""
    def decorator(func):
        _REGISTRY[name] = {
            "func": func,
            "group": group,
            "name": name,
        }
        return func
    return decorator


def get_registry() -> dict[str, dict]:
    return _REGISTRY.copy()


def get_feature_names() -> list[str]:
    return list(_REGISTRY.keys())


def get_groups() -> dict[str, list[str]]:
    """Return {group_name: [feature_names]}."""
    groups: dict[str, list[str]] = {}
    for name, info in _REGISTRY.items():
        groups.setdefault(info["group"], []).append(name)
    return groups


def apply_features(df, feature_names: list[str]):
    """Apply a list of feature functions by name."""
    import pandas as pd
    for name in feature_names:
        if name not in _REGISTRY:
            raise ValueError(f"Unknown feature: {name}")
        df = _REGISTRY[name]["func"](df)
    return df


# Import all feature modules so they self-register
from processing.features import temporal
from processing.features import lags
from processing.features import holidays
from processing.features import oil
from processing.features import store
from processing.features import promotions
from processing.features import interactions
from processing.features import target_encoding
from processing.features import yearly

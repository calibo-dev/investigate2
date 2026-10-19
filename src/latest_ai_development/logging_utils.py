from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """
    Configure application logging once at startup.

    If handlers already exist, keep the current handler set and only update
    the effective log level so uvicorn and pytest logging behavior is preserved.
    """

    numeric_level = getattr(logging, (level or "INFO").strip().upper(), logging.INFO)
    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.setLevel(numeric_level)
        return

    logging.basicConfig(level=numeric_level, format=DEFAULT_LOG_FORMAT)


def summarize_state_keys(state: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(state, Mapping):
        return []

    return sorted(str(key) for key in state.keys())


def summarize_state_delta(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, list[str]]:
    if not isinstance(after, Mapping):
        return {
            "added": [],
            "removed": [],
            "changed": [],
        }

    before_map = before if isinstance(before, Mapping) else {}
    before_keys = {str(key) for key in before_map.keys()}
    after_keys = {str(key) for key in after.keys()}

    changed = sorted(
        key for key in before_keys & after_keys if before_map.get(key) != after.get(key)
    )

    return {
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changed": changed,
    }


def summarize_tools(tools: Sequence[Any] | None) -> dict[str, Any]:
    tool_names: list[str] = []
    for tool in tools or []:
        tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if not tool_name:
            tool_name = tool.__class__.__name__
        tool_names.append(str(tool_name))

    return {
        "count": len(tool_names),
        "names": tool_names,
    }

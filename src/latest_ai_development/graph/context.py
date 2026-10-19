from __future__ import annotations

from typing import Any, TypedDict


class RuntimeContext(TypedDict, total=False):
    """Per-invocation values available to nodes without persisting them in state."""

    values: dict[str, Any]

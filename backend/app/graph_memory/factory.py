"""Backend selection for MiroFish graph memory.

``GRAPH_MEMORY_BACKEND=zep`` (default) keeps the original Zep Cloud behaviour.
``GRAPH_MEMORY_BACKEND=mempalace`` uses the local MemPalace backend.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

SUPPORTED_BACKENDS = ("zep", "mempalace")

_mempalace_lock = threading.Lock()
_mempalace_client = None


def graph_memory_backend() -> str:
    value = (os.environ.get("GRAPH_MEMORY_BACKEND") or "zep").strip().lower()
    if value not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported GRAPH_MEMORY_BACKEND={value!r}; expected one of {SUPPORTED_BACKENDS}"
        )
    return value


def is_mempalace_backend() -> bool:
    return graph_memory_backend() == "mempalace"


def graph_memory_config_errors() -> list[str]:
    """Configuration problems for the selected backend (empty list = OK)."""

    from ..config import Config

    try:
        backend = graph_memory_backend()
    except ValueError as error:
        return [str(error)]
    errors: list[str] = []
    if backend == "zep":
        if not Config.ZEP_API_KEY:
            errors.append("ZEP_API_KEY 未配置 (or set GRAPH_MEMORY_BACKEND=mempalace)")
        if os.environ.get("ZEP_API_URL"):
            errors.append("ZEP_API_URL 不受支持；MiroFish 仅连接 Zep Cloud")
    return errors


def graph_memory_configured() -> bool:
    return not graph_memory_config_errors()


def mempalace_data_dir() -> str:
    default = os.path.join(os.path.dirname(__file__), "../../uploads/mempalace")
    return os.path.abspath(os.environ.get("MEMPALACE_DATA_DIR") or default)


def get_graph_memory_client(api_key: Optional[str] = None, timeout: Optional[float] = None) -> Any:
    """Return the process-shared client for the configured backend."""

    if graph_memory_backend() == "zep":
        from ..utils.zep import get_zep_client

        return get_zep_client(api_key, timeout)

    global _mempalace_client
    with _mempalace_lock:
        if _mempalace_client is None:
            from .mempalace_backend import MemPalaceGraphMemory

            _mempalace_client = MemPalaceGraphMemory(
                mempalace_data_dir(),
                extract_workers=int(os.environ.get("MEMPALACE_EXTRACT_WORKERS", "4")),
            )
        return _mempalace_client


def reset_graph_memory_client() -> None:
    """Testing hook: drop the cached MemPalace client."""

    global _mempalace_client
    with _mempalace_lock:
        _mempalace_client = None

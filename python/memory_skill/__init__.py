"""Memory Skill — lightweight, real-time, self-evolving Agent memory module."""

# Version is read dynamically from package metadata so it can never drift
# from pyproject.toml again (0.5.0 was hardcoded while 0.6.0/0.7.0 shipped).
try:
    from importlib.metadata import version as _metadata_version

    __version__ = _metadata_version("memory-skill")
except Exception:  # source checkout without install metadata
    __version__ = "0.7.1"

from memory_skill.contracts import (
    DialogueTurn,
    MemoryEntry,
    MemoryEnvelope,
    MemorySkillConfig,
    MemorySkillError,
    ModelLoadError,
    SawEntry,
    SearchStoreCorruptionError,
    StoreCorruptionError,
)
from memory_skill.skill import MemorySkill
from memory_skill.weaver import WeaveContext

# Lazy import — room_adapter depends on the Room package,
# which is NOT available in OpenCode/MCP environments.
try:
    from memory_skill.room_adapter import MemorySkillAdapter
except ImportError:
    MemorySkillAdapter = None  # type: ignore[assignment]

__all__ = [
    "MemorySkill",
    "MemorySkillConfig",
    "WeaveContext",
    "MemoryEntry",
    "MemoryEnvelope",
    "DialogueTurn",
    "SawEntry",
    "MemorySkillError",
    "StoreCorruptionError",
    "ModelLoadError",
    "SearchStoreCorruptionError",
    "MemorySkillAdapter",
]

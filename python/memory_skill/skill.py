"""Memory Skill — public API (backward-compatible re-export).

``MemorySkill`` is now ``MemorySystem`` from ``_compose``.
The old god-object class is replaced by a dataclass with public
store attributes.  All method signatures are preserved.
"""

from memory_skill._compose import MemorySystem, create

MemorySkill = MemorySystem


def __getattr__(name: str):
    if name == "MemorySkillConfig":
        from memory_skill.contracts import MemorySkillConfig as Cfg
        globals()["MemorySkillConfig"] = Cfg
        return Cfg
    raise AttributeError(name)

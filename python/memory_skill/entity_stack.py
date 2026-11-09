"""会话实体栈 + 代词消解（方案 C 的指代层）。

设计（与用户敲定）：
  * 窗口内：栈里最近 3 轮出现过的、类型兼容的实体 → 直接消解（confidence=high）
  * 窗口外：走模糊召回，返回候选让上层"问一句"（confidence=guess / blur / none）
  * 栈有 TTL（默认 20 分钟），避免隔天拿旧实体消解

类型推断（从事实卡的 key 来）：
  key == name / user_name        -> self（用户本人）
  含 cat / pet / dog / bird      -> pet
  含 friend / colleague / boss   -> person
  其它                            -> thing
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# 代词 → 期望的实体类型（"我""你"是对话角色，不做消解）
PRONOUN_TYPES: dict[str, str] = {
    "他": "person", "她": "person", "他们": "person", "她们": "person",
    "它": "pet", "它们": "pet",
}
# 兼容：pet 实体也能被"他/她"指代（很多人这么说话）
TYPE_COMPAT: dict[str, set[str]] = {
    "person": {"person"},
    "pet": {"pet", "thing"},
    "thing": {"thing"},
    "self": {"self"},
}


@dataclass
class EntityRef:
    entity_id: str
    name: str
    etype: str
    turn_id: str | None = None
    ts: float = field(default_factory=time.time)
    dist: int = 0          # 距当前多少轮


def infer_type(key: str) -> str:
    k = (key or "").lower()
    if k in ("name", "user_name", "my_name"):
        return "self"
    if any(w in k for w in ("cat", "pet", "dog", "bird", "fish")):
        return "pet"
    if any(w in k for w in ("friend", "colleague", "boss", "partner", "person")):
        return "person"
    return "thing"


class EntityStack:
    """进程内实体栈：push 已知实体，resolve 代词。"""

    def __init__(self, window: int = 3, ttl_seconds: int = 1200,
                 max_size: int = 50) -> None:
        self._window = window
        self._ttl = ttl_seconds
        self._max = max_size
        self._items: list[EntityRef] = []
        self._turn_counter = 0

    # ── 栈操作 ────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """每轮对话开始时调用一次，推进所有实体的距离并清理过期项。"""
        now = time.time()
        for item in self._items:
            item.dist += 1
        self._items = [i for i in self._items if now - i.ts <= self._ttl][-self._max:]

    def push(self, entity_id: str, name: str, etype: str,
             turn_id: str | None = None) -> None:
        if not entity_id or not name:
            return
        now = time.time()
        # 已有同一实体 → 提到栈顶（距离由 tick 统一推进）
        self._items = [i for i in self._items if i.entity_id != entity_id]
        self._items.append(
            EntityRef(entity_id=entity_id, name=name, etype=etype,
                      turn_id=turn_id, ts=now, dist=0)
        )
        self._items = [i for i in self._items if now - i.ts <= self._ttl][-self._max:]

    def clear(self) -> None:
        self._items.clear()

    def snapshot(self) -> list[dict]:
        return [
            {"entity_id": i.entity_id, "name": i.name,
             "type": i.etype, "dist": i.dist}
            for i in reversed(self._items)
        ]

    # ── 消解 ──────────────────────────────────────────────────────────────

    @staticmethod
    def find_pronouns(text: str) -> list[tuple[str, str]]:
        """返回 [(代词, 期望类型)]，长代词优先（"他们"先于"他"）。"""
        found: list[tuple[str, str]] = []
        for word in sorted(PRONOUN_TYPES, key=len, reverse=True):
            if re.search(re.escape(word), text):
                kind = PRONOUN_TYPES[word]
                if not any(k == kind for _, k in found):
                    found.append((word, kind))
        return found

    def resolve(self, text: str) -> dict | None:
        """窗口内消解；找不到返回 None（由上层走模糊召回）。"""
        pronouns = self.find_pronouns(text)
        if not pronouns:
            return None
        now = time.time()
        for word, kind in pronouns:
            for item in reversed(self._items):
                if now - item.ts > self._ttl:
                    continue
                if item.dist > self._window:
                    continue
                if item.etype not in TYPE_COMPAT.get(kind, {kind}):
                    continue
                return {
                    "pronoun": word,
                    "entity_id": item.entity_id,
                    "name": item.name,
                    "type": item.etype,
                    "confidence": "high",
                    "dist": item.dist,
                }
        return None

    def candidates(self, limit: int = 3) -> list[dict]:
        """窗口外的候选（用于"我好像记得你提过某人"这类模糊话术）。"""
        now = time.time()
        out = []
        for item in reversed(self._items):
            if now - item.ts > self._ttl:
                continue
            out.append({
                "entity_id": item.entity_id,
                "name": item.name,
                "type": item.etype,
                "dist": item.dist,
                "title": f"{item.name}（{item.dist} 轮前）",
            })
            if len(out) >= limit:
                break
        return out

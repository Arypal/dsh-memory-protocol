"""跨语种别名索引。

目的：让英文名/别称（Daju）能精确命中中文条目（大橘），不依赖向量相似度。
中文模型的中英跨语言相似度只有 0.6 左右，落在检索门槛之下，别名匹配可以
绕过这个限制。

数据与 dialogue_turns 同库（memory.db），一张极小的表：

    entry_aliases(alias TEXT, entry_id TEXT, PRIMARY KEY(alias, entry_id))

匹配规则：
  * 中文别名：子串包含（无词边界概念）
  * 英文别名：大小写不敏感 + 词边界，避免 "cat" 命中 "concatenate"
  * 长别名优先，同一 entry 只返回一次
"""

from __future__ import annotations

import re
import sqlite3
import threading

_DDL = """
CREATE TABLE IF NOT EXISTS entry_aliases (
    alias    TEXT NOT NULL,
    entry_id TEXT NOT NULL,
    scope    TEXT NOT NULL DEFAULT 'web',
    PRIMARY KEY (alias, entry_id)
);
"""


class AliasIndex:
    """SQLite-backed alias → entry_id index with in-memory hot cache."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._map: dict[str, list[str]] | None = None
        self._ensure_schema()

    # ── internals ─────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_schema(self) -> None:
        try:
            conn = self._connect()
            try:
                conn.executescript(_DDL)
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass  # 只读/被占用时报错不应影响检索主流程
        # 记忆分区：老库补 scope 列（幂等）
        try:
            conn = self._connect()
            try:
                conn.execute(
                    "ALTER TABLE entry_aliases ADD COLUMN scope TEXT NOT NULL DEFAULT 'web'"
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    def _load(self) -> dict[str, list[tuple[str, str]]]:
        if self._map is not None:
            return self._map
        mapping: dict[str, list[tuple[str, str]]] = {}
        try:
            conn = self._connect()
            try:
                for alias, entry_id, scope in conn.execute(
                    "SELECT alias, entry_id, scope FROM entry_aliases"
                ):
                    mapping.setdefault(alias, []).append((entry_id, scope))
            finally:
                conn.close()
        except Exception:
            mapping = {}
        self._map = mapping
        return mapping

    # ── public API ────────────────────────────────────────────────────────

    def add(self, alias: str, entry_id: str, scope: str = "web") -> None:
        alias = (alias or "").strip()
        if len(alias) < 2 or not entry_id:
            return
        with self._lock:
            try:
                conn = self._connect()
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO entry_aliases(alias, entry_id, scope) "
                        "VALUES (?, ?, ?)",
                        (alias, entry_id, scope),
                    )
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                return
            self._map = None

    def add_many(self, aliases: list[str], entry_id: str, scope: str = "web") -> int:
        count = 0
        for alias in aliases:
            self.add(alias, entry_id, scope)
            count += 1
        return count

    def remove(self, alias: str, entry_id: str | None = None) -> None:
        with self._lock:
            try:
                conn = self._connect()
                try:
                    if entry_id:
                        conn.execute(
                            "DELETE FROM entry_aliases WHERE alias=? AND entry_id=?",
                            (alias, entry_id),
                        )
                    else:
                        conn.execute(
                            "DELETE FROM entry_aliases WHERE alias=?", (alias,)
                        )
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                return
            self._map = None

    def list_all(self, scope: str | None = None) -> list[tuple[str, str]]:
        return [
            (alias, eid)
            for alias, pairs in self._load().items()
            for eid, sc in pairs
            if scope is None or sc == scope
        ]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """ASCII 单词集合（小写）；中文交给子串匹配。"""
        return set(re.findall(r"[a-z0-9]+", text))

    def hits(self, text: str, scope: str | None = None) -> list[str]:
        """Return entry ids whose alias occurs in *text*, longest alias first.

        ``scope=None`` searches every partition (Web cross-partition reads);
        a concrete scope (e.g. ``"platform"``) restricts results to it.

        0.4.2: ASCII aliases also match by token coverage (≥2/3), so a question
        about ``dsh-memory-protocol`` reaches the card keyed
        ``memory_protocol_plugin`` even though the alias never appears
        verbatim. Exact matches rank before fuzzy ones.
        """
        if not text:
            return []
        low = text.lower()
        query_tokens = self._tokens(low)
        mapping = self._load()
        exact: list[str] = []
        fuzzy: list[str] = []
        for alias in sorted(mapping, key=len, reverse=True):
            a = alias.lower()
            if a.isascii():
                if re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", low):
                    exact.append(alias)
                    continue
                toks = self._tokens(a)
                if len(toks) >= 2:
                    covered = sum(1 for t in toks if t in query_tokens)
                    if covered / len(toks) >= 0.66:
                        fuzzy.append(alias)
            else:
                if a in low:
                    exact.append(alias)
        out: list[str] = []
        seen: set[str] = set()
        for alias in exact + fuzzy:
            for entry_id, entry_scope in mapping[alias]:
                if scope is not None and entry_scope != scope:
                    continue
                if entry_id not in seen:
                    seen.add(entry_id)
                    out.append(entry_id)
        return out

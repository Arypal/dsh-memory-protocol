"""事实卡存储（方案 C 的 L1 + L2）。

L1 facts         : 结论，key-value，可覆盖，值变化进 history
L2 fact_evidence : 完整原话 + 时间戳，只追加

设计要点：
  * 证据存原文副本而不是外键，删掉对话也不影响时间线
  * 建卡时自动生成双语别名（中文值 + 英文 key 变体），写入 entry_aliases
  * 事实卡同时进向量库（id 前缀 fact:），这样语义检索和自动 mention 都能命中
"""

from __future__ import annotations

import difflib
import sqlite3
import threading
import time

from memory_skill import secure

_DDL = """
CREATE TABLE IF NOT EXISTS facts (
    id            TEXT PRIMARY KEY,
    category      TEXT NOT NULL,
    key           TEXT NOT NULL,
    current_value TEXT NOT NULL,
    confidence    REAL DEFAULT 0.8,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    -- 生命周期状态：active（有效）/ done（已被取代或已完成）
    status        TEXT DEFAULT 'active',
    superseded_by TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_cat_key ON facts(category, key);

CREATE TABLE IF NOT EXISTS fact_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id     TEXT NOT NULL,
    value       TEXT NOT NULL,
    since       REAL NOT NULL,
    until       REAL,
    evidence_id TEXT
);

CREATE TABLE IF NOT EXISTS fact_evidence (
    id      TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL,
    type    TEXT NOT NULL,
    quote   TEXT NOT NULL,
    said_at REAL NOT NULL,
    turn_id TEXT,
    speaker TEXT,
    -- 说话时刻（来自被引用 turn 的 timestamp），与 said_at（写入时刻）分开存。
    -- 回填/迁移写入时 said_at 会偏离真实说话时间，展示与排序一律优先用它。
    source_said_at REAL
);
CREATE INDEX IF NOT EXISTS idx_ev_fact ON fact_evidence(fact_id, said_at DESC);

-- 四域作用域：会话 → 绑定的域（world 永远可见，world 之外按绑定取用）
CREATE TABLE IF NOT EXISTS scope_bindings (
    session_id TEXT NOT NULL,
    scope      TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, scope)
);
"""


def build_aliases(key: str, value: str, extra: list[str] | None = None) -> list[str]:
    """自动生成双语别名：中文值原样 + 英文 key 的几种写法 + 显式别名。"""
    candidates = [
        value,
        key,
        key.replace("_", " "),
        key.replace(" ", "_"),
        *(extra or []),
    ]
    out: list[str] = []
    for item in candidates:
        item = (item or "").strip()
        if len(item) >= 2 and item not in out:
            out.append(item)
    return out


# ── 事实卡生命周期：相似去重 ────────────────────────────────────────────
# 只有"会更新/会过时"的类别参与自动取代；身份与偏好（pers/pref）一旦写下
# 就不该被另一张卡悄悄顶掉。
_DEDUP_CATEGORIES = {"topic", "lesson", "conclusion", "context"}
_DEDUP_THRESHOLD = 0.80


def similarity(a: str, b: str) -> float:
    """两张卡的文本相似度：字符 bigram Jaccard 与序列相似度取较大者。

    短于 8 字符的文本只做相等判断，避免"已完成"/"修好了"这类短值被误合并。
    """
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) < 8 or len(b) < 8:
        return 0.0
    grams_a = {a[i:i + 2] for i in range(len(a) - 1)}
    grams_b = {b[i:i + 2] for i in range(len(b) - 1)}
    jaccard = len(grams_a & grams_b) / max(1, len(grams_a | grams_b))
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return max(jaccard, ratio)


class FactStore:
    """SQLite 事实卡 + 证据时间线，并与向量库、别名表保持同步。"""

    def __init__(self, db_path: str, alias_index=None, learned_store=None) -> None:
        self._db_path = db_path
        self._alias_index = alias_index
        self._learned = learned_store
        self._lock = threading.Lock()
        self._ensure_schema()

    # ── internals ─────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_schema(self) -> None:
        try:
            conn = self._conn()
            try:
                conn.executescript(_DDL)
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass
        # 列级升级（旧库）；ALTER 重复执行会抛错，忽略即可（幂等）
        for ddl in (
            "ALTER TABLE fact_evidence ADD COLUMN secondary INTEGER DEFAULT 0",
            "ALTER TABLE fact_evidence ADD COLUMN source_said_at REAL",
            "ALTER TABLE facts ADD COLUMN scope TEXT NOT NULL DEFAULT 'web'",
            "ALTER TABLE facts ADD COLUMN status TEXT DEFAULT 'active'",
            "ALTER TABLE facts ADD COLUMN superseded_by TEXT",
        ):
            try:
                conn = self._conn()
                try:
                    conn.execute(ddl)
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                pass

    @staticmethod
    def _as_epoch(value) -> float | None:
        """把 float / datetime 归一成 unix 秒。"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value.timestamp())
        except Exception:
            return None

    def _turn_time(self, turn_id: str | None) -> float | None:
        """从对话流水取被引用 turn 的真实说话时刻（找不到返回 None）。"""
        if not turn_id:
            return None
        try:
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT timestamp FROM dialogue_turns WHERE id=?", (turn_id,)
                ).fetchone()
            finally:
                conn.close()
        except Exception:
            return None
        return float(row[0]) if row else None

    def _source_time(self, turn_id: str | None, source_said_at=None) -> float | None:
        """说话时刻：显式传入优先，否则从 turn_id 反查；都没有就留空。"""
        explicit = self._as_epoch(source_said_at)
        if explicit is not None:
            return explicit
        return self._turn_time(turn_id)

    def _sync_vector(self, fact_id: str, content: str, category: str,
                     created_at: float, scope: str = "web") -> None:
        """把事实卡同步进向量库，供语义检索与自动 mention 使用。"""
        if self._learned is None:
            return
        try:
            from datetime import UTC, datetime

            from memory_skill.contracts import MemoryEntry

            ts = datetime.fromtimestamp(created_at, UTC)
            entry = MemoryEntry(
                id=fact_id,
                content=content,
                created_at=ts,
                updated_at=ts,
                weight=0.5 if category in ("context", "lesson", "topic") else 0.9,
                category=category,
                tags=["fact"],
                metadata={"source": "fact", "category": category, "scope": scope},
            )
            existing = self._learned._collection.get(ids=[fact_id])
            if existing.get("ids"):
                self._learned.update(fact_id, content=content)
            else:
                self._learned.insert(entry)
        except Exception:
            pass

    def _register_aliases(self, fact_id: str, aliases: list[str],
                          scope: str = "web") -> None:
        if self._alias_index is None:
            return
        for alias in aliases:
            try:
                self._alias_index.add(alias, fact_id, scope)
            except Exception:
                pass

    def _supersede_similar(self, category: str, key: str, value: str,
                           new_id: str, scope: str = "web") -> list[dict]:
        """新增卡时，把同类别同分区里"已过时/高度重复"的旧卡标记为 done。

        旧卡不删数据：``status='done'`` + ``superseded_by`` 指向新卡，
        证据时间线保留；但向量库与别名表里的条目会被移除，否则语义检索
        仍会命中已过时的说法（这是"新卡挤掉旧卡"的根因）。
        """
        if category not in _DEDUP_CATEGORIES:
            return []
        try:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT id, key, current_value FROM facts "
                    "WHERE category=? AND scope=? AND key!=? "
                    "AND COALESCE(status, 'active')='active'",
                    (category, scope, key),
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            return []

        best = None
        for old_id, old_key, old_value in rows:
            try:
                plain = secure.decrypt(old_value or "")
            except Exception:
                plain = ""
            score = similarity(value, plain)
            if score >= _DEDUP_THRESHOLD and (best is None or score > best[0]):
                best = (score, old_id, old_key, plain)
        if best is None:
            return []

        score, old_id, old_key, old_value = best
        try:
            conn = self._conn()
            try:
                conn.execute(
                    "UPDATE facts SET status='done', superseded_by=? WHERE id=?",
                    (new_id, old_id),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            return []

        try:
            if self._learned is not None:
                self._learned.delete(old_id)
        except Exception:
            pass
        if self._alias_index is not None:
            for alias in build_aliases(old_key, old_value or ""):
                try:
                    self._alias_index.remove(alias, old_id)
                except Exception:
                    pass

        return [{"id": old_id, "key": old_key,
                 "score": round(score, 3), "value": old_value}]

    def set_status(self, key_or_id: str, status: str = "done",
                   scope: str = "web") -> dict:
        """手动改卡的生命周期状态（done 后不再进注入 / 检索 / 列表）。

        done 时同步把该卡从向量库和别名表移除，否则语义检索仍会命中；
        active 时把向量卡与别名重建回来（可撤销）。
        """
        status = (status or "done").strip().lower()
        if status not in {"active", "done"}:
            return {"status": "error", "reason": "status must be active|done"}
        target = (key_or_id or "").strip()
        if not target:
            return {"status": "error", "reason": "key required"}
        is_id = target.startswith("fact:")
        try:
            conn = self._conn()
            try:
                if is_id:
                    rows = conn.execute(
                        "SELECT id, key, category, current_value, created_at "
                        "FROM facts WHERE id=?", (target,)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT id, key, category, current_value, created_at "
                        "FROM facts WHERE key=? AND scope=?", (target, scope)).fetchall()
                if not rows:
                    return {"status": "error", "reason": "fact not found"}
                if is_id:
                    conn.execute(
                        "UPDATE facts SET status=? WHERE id=?", (status, target))
                else:
                    conn.execute(
                        "UPDATE facts SET status=? WHERE key=? AND scope=?",
                        (status, target, scope))
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "reason": str(exc)}

        for fid, key, category, value, created_at in rows:
            if status == "done":
                try:
                    if self._learned is not None:
                        self._learned.delete(fid)
                except Exception:
                    pass
                if self._alias_index is not None:
                    for alias in build_aliases(key, value or ""):
                        try:
                            self._alias_index.remove(alias, fid)
                        except Exception:
                            pass
            else:
                rich = f"{key.replace(chr(95), chr(32))}: {value}".strip()
                self._sync_vector(fid, rich, category,
                                  float(created_at or time.time()), scope)
                self._register_aliases(fid, build_aliases(key, value or ""), scope)
        return {"status": "ok", "changed": len(rows), "new_status": status}

    # ── 四域作用域（world / topic: / file: / session:）──────────────────

    def bind_scope(self, session_id: str, scope: str) -> dict:
        """把某个域绑到会话上（读/写命中即绑定，参考实现的自动绑定语义）。"""
        session_id = (session_id or "web").strip() or "web"
        scope = (scope or "").strip()
        if not scope:
            return {"status": "error", "reason": "scope required"}
        try:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO scope_bindings"
                    "(session_id, scope, created_at) VALUES (?,?,?)",
                    (session_id, scope, time.time()),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "reason": str(exc)}
        return {"status": "ok", "session_id": session_id, "scope": scope}

    def list_scopes(self, session_id: str = "web") -> list[str]:
        """会话已绑定的域（不含 world）。"""
        try:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT scope FROM scope_bindings WHERE session_id=? "
                    "ORDER BY created_at", (session_id,)).fetchall()
            finally:
                conn.close()
        except Exception:
            return []
        return [r[0] for r in rows]

    def visible_scopes(self, session_id: str = "web") -> list[str]:
        """检索可见的域集合：world 永远可见 + 会话自身 + 所有绑定域。"""
        scopes = {session_id or "web", "world"}
        scopes.update(self.list_scopes(session_id or "web"))
        return sorted(scopes)

    # ── public API ────────────────────────────────────────────────────────

    def upsert(
        self,
        category: str,
        key: str,
        value: str,
        aliases: list[str] | None = None,
        quote: str = "",
        turn_id: str | None = None,
        speaker: str = "user",
        evidence_type: str = "declaration",
        source_said_at=None,
        scope: str = "web",
    ) -> dict:
        """建立/更新一张事实卡，并追加一条证据。"""
        category = (category or "pers").strip()
        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return {"status": "error", "reason": "key/value required"}

        fact_id = (
            f"fact:{scope}:{category}:{key}" if scope != "web"
            else f"fact:{category}:{key}"
        )
        now = time.time()
        auto_aliases = build_aliases(key, value, aliases)
        is_new = False

        with self._lock:
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT id, current_value, created_at FROM facts "
                    "WHERE category=? AND key=? AND scope=?",
                    (category, key, scope),
                ).fetchone()

                if row is None:
                    is_new = True
                    conn.execute(
                        "INSERT INTO facts(id, category, key, current_value, "
                        "confidence, created_at, updated_at, scope) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (fact_id, category, key, secure.encrypt(value), 0.8,
                         now, now, scope),
                    )
                    conn.execute(
                        "INSERT INTO fact_history(fact_id, value, since) VALUES (?,?,?)",
                        (fact_id, secure.encrypt(value), now),
                    )
                    created_at = now
                    changed = True
                else:
                    created_at = float(row[2])
                    try:
                        old_value = secure.decrypt(row[1] or "")
                    except Exception:
                        old_value = None  # 读不出（篡改/换密钥）→ 视为变化
                    changed = old_value != value
                    if changed:
                        conn.execute(
                            "UPDATE fact_history SET until=? WHERE fact_id=? AND until IS NULL",
                            (now, fact_id),
                        )
                        conn.execute(
                            "UPDATE facts SET current_value=?, updated_at=? WHERE id=?",
                            (secure.encrypt(value), now, fact_id),
                        )
                        conn.execute(
                            "INSERT INTO fact_history(fact_id, value, since) VALUES (?,?,?)",
                            (fact_id, secure.encrypt(value), now),
                        )

                evidence_id = f"ev_{int(now * 1000)}_{abs(hash(quote)) % 10000:04d}"
                src_time = self._source_time(turn_id, source_said_at)
                conn.execute(
                    "INSERT OR REPLACE INTO fact_evidence"
                    "(id, fact_id, type, quote, said_at, turn_id, speaker, "
                    "source_said_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (evidence_id, fact_id, evidence_type,
                     secure.encrypt(quote or value),
                     now, turn_id, speaker, src_time),
                )
                conn.commit()
            finally:
                conn.close()

        # `value` is already the subject of the first segment; build_aliases
        # also lists it, which used to duplicate the whole value inside the
        # vector document and skew the card's embedding toward its own text.
        # 新卡写入后，自动取代同类别同分区里被它覆盖的旧卡（仅 topic / lesson /
        # conclusion / context；pers / pref 是长期身份，不做自动取代）。
        superseded: list[dict] = []
        if is_new:
            superseded = self._supersede_similar(
                category, key, value, fact_id, scope)

        _alias_text = " ".join(a for a in auto_aliases if a != value)
        rich = f"{key.replace(chr(95), chr(32))}: {value} | {_alias_text}".strip()
        self._sync_vector(fact_id, rich, category, created_at, scope)
        self._register_aliases(fact_id, auto_aliases, scope)

        # 四域：写进非默认域时把该域绑到当前会话（读写即绑定）。
        if scope not in ("web", "world"):
            try:
                self.bind_scope("web", scope)
            except Exception:
                pass

        return {
            "status": "ok",
            "fact_id": fact_id,
            "category": category,
            "key": key,
            "value": value,
            "changed": changed,
            "aliases": auto_aliases,
            "evidence_id": evidence_id,
            "superseded": superseded,
        }

    def append_mention(self, fact_id: str, quote: str,
                       turn_id: str | None = None,
                       speaker: str = "user",
                       secondary: bool = False,
                       source_said_at=None) -> bool:
        """追加一条 mention 证据（不改变事实值）。"""
        now = time.time()
        evidence_id = f"ev_{int(now * 1000)}_{abs(hash(quote)) % 10000:04d}"
        try:
            conn = self._conn()
            try:
                src_time = self._source_time(turn_id, source_said_at)
                conn.execute(
                    "INSERT OR REPLACE INTO fact_evidence"
                    "(id, fact_id, type, quote, said_at, turn_id, speaker, "
                    "secondary, source_said_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (evidence_id, fact_id, "mention", quote, now, turn_id,
                     speaker, 1 if secondary else 0, src_time),
                )
                conn.commit()
            finally:
                conn.close()
            return True
        except Exception:
            return False

    def get(self, fact_id: str) -> dict | None:
        try:
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT id, category, key, current_value, confidence, "
                    "created_at, updated_at, COALESCE(status, 'active'), "
                    "superseded_by FROM facts WHERE id=?",
                    (fact_id,),
                ).fetchone()
            finally:
                conn.close()
        except Exception:
            return None
        if not row:
            return None
        return {
            "id": row[0], "category": row[1], "key": row[2],
            "current_value": secure.decrypt_or_placeholder(row[3])[0],
            "confidence": row[4],
            "created_at": row[5], "updated_at": row[6], "status": row[7],
            "superseded_by": row[8],
        }

    def find_by_key(self, key: str, category: str | None = None,
                    scope: str = "web") -> dict | None:
        try:
            conn = self._conn()
            try:
                if category:
                    row = conn.execute(
                        "SELECT id FROM facts WHERE key=? AND category=? AND scope=?",
                        (key, category, scope),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT id FROM facts WHERE key=? AND scope=? LIMIT 1",
                        (key, scope),
                    ).fetchone()
            finally:
                conn.close()
        except Exception:
            return None
        return self.get(row[0]) if row else None

    def list_facts(self, category: str | None = None,
                   scope: str = "web",
                   include_inactive: bool = False,
                   scopes: list[str] | None = None) -> list[dict]:
        """列出事实卡；默认只返回 active（done / 已被取代的不进注入与检索）。

        ``scopes`` 给定时按多域可见（world + 绑定域），否则按单域 ``scope``。
        """
        clauses: list[str] = []
        params: list = []
        if scopes:
            clauses.append("scope IN (" + ",".join("?" * len(scopes)) + ")")
            params.extend(scopes)
        else:
            clauses.append("scope=?")
            params.append(scope)
        if category:
            clauses.append("category=?")
            params.append(category)
        if not include_inactive:
            clauses.append("COALESCE(status, 'active')='active'")
        where = " WHERE " + " AND ".join(clauses)
        try:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT id, category, key, current_value, updated_at, "
                    "COALESCE(status, 'active'), superseded_by "
                    "FROM facts" + where + " ORDER BY updated_at DESC",
                    params,
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            return []
        return [
            {"id": r[0], "category": r[1], "key": r[2],
             "current_value": secure.decrypt_or_placeholder(r[3])[0],
             "updated_at": r[4],
             "status": r[5], "superseded_by": r[6]}
            for r in rows
        ]

    def timeline(self, fact_id: str, limit: int = 50) -> list[dict]:
        try:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT id, type, quote, said_at, turn_id, speaker, "
                    "COALESCE(secondary, 0), source_said_at FROM fact_evidence "
                    "WHERE fact_id=? "
                    "ORDER BY COALESCE(source_said_at, said_at) DESC LIMIT ?",
                    (fact_id, limit),
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            return []
        return [
            {"id": r[0], "type": r[1],
             "quote": secure.decrypt_or_placeholder(r[2])[0],
             "said_at": r[3], "turn_id": r[4], "speaker": r[5],
             "secondary": bool(r[6]),
             "source_said_at": r[7],
             "effective_said_at": r[7] if r[7] is not None else r[3]}
            for r in rows
        ]

    def history(self, fact_id: str) -> list[dict]:
        try:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT value, since, until FROM fact_history "
                    "WHERE fact_id=? ORDER BY since",
                    (fact_id,),
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            return []
        return [{"value": secure.decrypt_or_placeholder(r[0])[0],
                 "since": r[1], "until": r[2]} for r in rows]

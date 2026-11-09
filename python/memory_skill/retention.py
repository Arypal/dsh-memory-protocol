"""L3 对话保留策略 + L4 对话摘要。

L3：对话流水有寿命。超过 TTL 或条数上限的旧对话会被清理。
L4：清理前先把这段区间压成一条 episode_summary，保留"那段时间在忙什么"的轮廓。

摘要生成有两档：
  * llm  —— 配了 IMPORTANCE_API_KEY 时调用 OpenAI 兼容接口
  * rule —— 没有 key 时用机械摘要（时间范围 + 涉及实体 + 高频词）

证据时间线（fact_evidence）存的是原文副本，清理对话不影响它。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import urllib.request
from collections import Counter
from datetime import datetime

_DDL = """
CREATE TABLE IF NOT EXISTS episode_summaries (
    id           TEXT PRIMARY KEY,
    period_start REAL NOT NULL,
    period_end   REAL NOT NULL,
    summary      TEXT NOT NULL,
    source_turns INTEGER NOT NULL,
    method       TEXT NOT NULL,
    generated_at REAL NOT NULL
);

-- 回收站：被 prune 的对话先在这里留 7 天，可恢复；到期才永久删除。
CREATE TABLE IF NOT EXISTS trash_turns (
    id         TEXT PRIMARY KEY,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    timestamp  REAL NOT NULL,
    deleted_at REAL NOT NULL,
    reason     TEXT
);
CREATE INDEX IF NOT EXISTS idx_trash_deleted ON trash_turns(deleted_at);
"""

_TRASH_TTL_DAYS = 7

_CN_WORD = re.compile(r"[\u4e00-\u9fff]{2,6}")
_STOP = {
    "这个", "那个", "我们", "你们", "他们", "什么", "怎么", "可以", "就是",
    "然后", "因为", "所以", "但是", "如果", "还是", "一个", "现在", "已经",
    "没有", "不是", "我的", "你的", "他的", "自己", "一下", "这样", "那样",
}


class RetentionManager:
    """对话清理（L3）与区间摘要（L4）。"""

    def __init__(self, db_path: str, dialogue_store=None, learned_store=None,
                 alias_index=None, config=None) -> None:
        self._db_path = db_path
        self._dialogue = dialogue_store
        self._learned = learned_store
        self._alias = alias_index
        self._config = config
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

    # ── L4：摘要 ──────────────────────────────────────────────────────────

    def _llm_summary(self, text: str) -> str | None:
        """有 key 时用 LLM 摘要；任何失败都返回 None，回落到规则摘要。"""
        api_key = os.environ.get("IMPORTANCE_API_KEY", "")
        if not api_key:
            return None
        base = os.environ.get("IMPORTANCE_API_BASE", "https://api.deepseek.com/v1")
        model = os.environ.get("IMPORTANCE_MODEL", "deepseek-chat")
        prompt = (
            "把下面这段对话压成一段不超过 80 字的中文摘要，"
            "只保留：主题、涉及的人/物、关键结论。不要评价，不要列条目。\n\n"
            + text[:6000]
        )
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(
            f"{base.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return (body["choices"][0]["message"]["content"] or "").strip() or None
        except Exception:
            return None

    def _rule_summary(self, turns: list) -> str:
        start = datetime.fromtimestamp(min(t[3] for t in turns))
        end = datetime.fromtimestamp(max(t[3] for t in turns))
        text = "\n".join((t[2] or "") for t in turns)
        names: list[str] = []
        if self._alias is not None:
            try:
                for alias, _eid in self._alias.list_all():
                    if len(alias) >= 2 and alias in text and alias not in names:
                        names.append(alias)
            except Exception:
                pass
        words = Counter(
            w for w in _CN_WORD.findall(text) if w not in _STOP
        ).most_common(5)
        parts = [f"{start:%m-%d %H:%M} ~ {end:%m-%d %H:%M}，{len(turns)} 轮对话"]
        if names:
            parts.append("涉及 " + "、".join(names[:6]))
        if words:
            parts.append("关键词 " + "、".join(w for w, _ in words))
        return "；".join(parts)

    def summarize(self, turns: list) -> dict | None:
        if not turns:
            return None
        start = min(t[3] for t in turns)
        end = max(t[3] for t in turns)
        text = "\n".join((t[2] or "") for t in turns)
        summary = self._llm_summary(text)
        method = "llm" if summary else "rule"
        if not summary:
            summary = self._rule_summary(turns)
        episode_id = f"ep_{int(start)}_{int(end)}"
        try:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO episode_summaries"
                    "(id, period_start, period_end, summary, source_turns, "
                    "method, generated_at) VALUES (?,?,?,?,?,?,?)",
                    (episode_id, start, end, summary, len(turns), method, time.time()),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            return None
        return {"id": episode_id, "summary": summary,
                "method": method, "turns": len(turns)}

    # ── L3：清理 ──────────────────────────────────────────────────────────

    def prune(self, ttl_days: int | None = None, max_turns: int | None = None,
              dry_run: bool = False) -> dict:
        cfg = self._config
        ttl = ttl_days if ttl_days is not None else getattr(
            cfg, "dialogue_ttl_days", 90)
        cap = max_turns if max_turns is not None else getattr(
            cfg, "dialogue_max_turns", 2000)

        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT id, role, content, timestamp FROM dialogue_turns "
                "ORDER BY timestamp ASC"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {"status": "ok", "deleted": 0, "kept": 0,
                    "reason": "empty"}

        cutoff = time.time() - ttl * 86400 if ttl > 0 else 0
        by_age = [r for r in rows if r[3] < cutoff]
        remaining = [r for r in rows if r[3] >= cutoff]
        by_cap: list = []
        if cap > 0 and len(remaining) > cap:
            overflow = len(remaining) - cap
            by_cap = remaining[:overflow]
            remaining = remaining[overflow:]

        victims = by_age + by_cap
        if not victims:
            return {"status": "ok", "deleted": 0, "kept": len(rows),
                    "reason": "nothing to prune"}

        episode = self.summarize(victims)
        if dry_run:
            return {"status": "dry_run", "would_delete": len(victims),
                    "kept": len(remaining),
                    "episode": episode}

        # Fail closed: if the episode summary could not be persisted, deleting
        # the turns would lose that stretch of history outright. fish-memory
        # keeps everything and only down-weights it; this is the equivalent
        # guarantee at this layer — nothing is removed until its summary is
        # safely stored.
        if episode is None:
            return {"status": "skipped", "deleted": 0, "kept": len(rows),
                    "reason": "summary_write_failed"}

        ids = [r[0] for r in victims]
        placeholders = ",".join("?" * len(ids))
        now = time.time()
        conn = self._conn()
        try:
            # 先入回收站（7 天内可恢复），再删主表
            conn.executemany(
                "INSERT OR REPLACE INTO trash_turns"
                "(id, role, content, timestamp, deleted_at, reason) "
                "VALUES (?,?,?,?,?,?)",
                [(r[0], r[1], r[2], r[3], now, "prune") for r in victims],
            )
            conn.execute(
                f"DELETE FROM dialogue_turns WHERE id IN ({placeholders})", ids)
            conn.execute(
                f"DELETE FROM dialogue_fts WHERE turn_id IN ({placeholders})", ids)
            conn.commit()
        finally:
            conn.close()

        trashed = len(victims)
        trash_purged = self.purge_trash()

        removed_vectors = 0
        if self._learned is not None:
            for tid in ids:
                try:
                    self._learned.delete(f"dialogue:{tid}")
                    removed_vectors += 1
                except Exception:
                    pass

        return {
            "status": "ok",
            "deleted": len(ids),
            "trashed": trashed,
            "trash_purged": trash_purged,
            "deleted_vectors": removed_vectors,
            "kept": len(remaining),
            "episode": episode,
        }

    # ── 回收站（trash：删掉的对话留 7 天，可恢复）──────────────────────

    def list_trash(self, limit: int = 20) -> list[dict]:
        """列出回收站里的对话（按删除时间倒序）。"""
        try:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT id, role, substr(content, 1, 200), timestamp, "
                    "deleted_at FROM trash_turns "
                    "ORDER BY deleted_at DESC LIMIT ?", (limit,)).fetchall()
            finally:
                conn.close()
        except Exception:
            return []
        return [{"id": r[0], "role": r[1], "preview": r[2],
                 "timestamp": r[3], "deleted_at": r[4]} for r in rows]

    def restore_from_trash(self, turn_ids: list[str] | None = None,
                           limit: int = 100) -> dict:
        """把回收站里的对话放回 dialogue_turns + FTS。

        向量不重建（下次检索走 BM25 仍能命中）；这是"误删恢复"的兜底通道。
        """
        rows: list = []
        conn = self._conn()
        try:
            if turn_ids:
                marks = ",".join("?" * len(turn_ids))
                rows = conn.execute(
                    f"SELECT id, role, content, timestamp FROM trash_turns "
                    f"WHERE id IN ({marks})", turn_ids).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, role, content, timestamp FROM trash_turns "
                    "ORDER BY deleted_at ASC LIMIT ?", (limit,)).fetchall()
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO dialogue_turns"
                    "(id, role, content, timestamp) VALUES (?,?,?,?)", rows)
                conn.executemany(
                    "INSERT OR REPLACE INTO dialogue_fts(turn_id, content) "
                    "VALUES (?,?)", [(r[0], r[2]) for r in rows])
                conn.executemany(
                    "DELETE FROM trash_turns WHERE id=?", [(r[0],) for r in rows])
                conn.commit()
        finally:
            conn.close()
        return {"status": "ok", "restored": len(rows)}

    def purge_trash(self, older_than_days: int | None = None) -> int:
        """永久删除回收站里超过 N 天的对话（默认 7 天）。返回删除条数。"""
        days = _TRASH_TTL_DAYS if older_than_days is None else older_than_days
        cutoff = time.time() - days * 86400
        conn = self._conn()
        try:
            cur = conn.execute(
                "DELETE FROM trash_turns WHERE deleted_at < ?", (cutoff,))
            conn.commit()
            return cur.rowcount or 0
        except Exception:
            return 0
        finally:
            conn.close()

    # ── 查询 ──────────────────────────────────────────────────────────────

    def list_episodes(self, limit: int = 20) -> list[dict]:
        try:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT id, period_start, period_end, summary, source_turns, "
                    "method FROM episode_summaries ORDER BY period_end DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            return []
        return [
            {"id": r[0], "period_start": r[1], "period_end": r[2],
             "summary": r[3], "source_turns": r[4], "method": r[5]}
            for r in rows
        ]

    def find_episode_for(self, ts: float) -> dict | None:
        for ep in self.list_episodes(200):
            if ep["period_start"] <= ts <= ep["period_end"]:
                return ep
        return None

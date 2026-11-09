"""
Memory Skill — TreeManager: SQLite-backed navigable memory tree.

V2: Auto-placed date→period hierarchy.  LLM only classifies {root, branch}.
Time layers (date, period) are system-managed.  Fixed five-level depth:

    root → branch → date → period → memory

    user_root (用户记忆根)
      ├── user_pref (偏好)
      │   ├── user_pref_2026-07-19 (2026-07-19)
      │   │   ├── user_pref_2026-07-19_am (上午)
      │   │   │   └── user_pref_2026-07-19_am_a1b2c3d4 (content…)
      │   │   └── user_pref_2026-07-19_pm (下午)
      │   │       └── user_pref_2026-07-19_pm_e5f6g7h8 (content…)
      └── user_mem (回忆与目标)
          └── ...
    assistant_root (助手记忆根)
      ├── assistant_pers (人格)
      └── assistant_task (任务与技能)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Any

_logger = logging.getLogger(__name__)

# ── Schema ──────────────────────────────────────────────────────────────────────

_CREATE_TREE_SQL = """
CREATE TABLE IF NOT EXISTS tree_nodes (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    root TEXT NOT NULL CHECK(root IN ('user', 'assistant')),
    label TEXT NOT NULL,
    level INTEGER NOT NULL DEFAULT 0,
    memory_ref TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

# ── Root / base branch definitions ──────────────────────────────────────────────

_ROOT_NODES: list[dict[str, Any]] = [
    {"id": "user_root", "parent_id": None, "root": "user", "label": "用户记忆根", "level": 0},
    {"id": "assistant_root", "parent_id": None, "root": "assistant", "label": "助手记忆根", "level": 0},
]

_BASE_BRANCHES: list[dict[str, Any]] = [
    # User branches
    {"id": "user_pref", "parent_id": "user_root", "root": "user", "label": "偏好", "level": 1},
    {"id": "user_mem", "parent_id": "user_root", "root": "user", "label": "回忆与目标", "level": 1},
    # Assistant branches
    {"id": "assistant_pers", "parent_id": "assistant_root", "root": "assistant", "label": "人格", "level": 1},
    {"id": "assistant_task", "parent_id": "assistant_root", "root": "assistant", "label": "任务", "level": 1},
    {"id": "assistant_skill", "parent_id": "assistant_root", "root": "assistant", "label": "技能", "level": 1},
]

# Branch ID → root lookup
_BRANCH_TO_ROOT: dict[str, str] = {b["id"]: b["root"] for b in _BASE_BRANCHES}

# Branch ID → Chinese label for display
_BRANCH_LABEL_MAP: dict[str, str] = {
    "user_pref": "偏好",
    "user_mem": "回忆与目标",
    "assistant_pers": "人格",
    "assistant_task": "任务",
    "assistant_skill": "技能",
}

# ── Period helpers ──────────────────────────────────────────────────────────────

_PERIOD_MAP: dict[str, str] = {
    "am": "上午",
    "pm": "下午",
    "ev": "晚上",
}

_PERIOD_HOUR_MAP: dict[str, tuple[int, ...]] = {
    "am": (6, 7, 8, 9, 10, 11),
    "pm": (12, 13, 14, 15, 16, 17),
    "ev": (18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5),
}


def _period_for_hour(hour: int) -> str:
    if 6 <= hour <= 11:
        return "am"
    elif 12 <= hour <= 17:
        return "pm"
    else:
        return "ev"


# ── Default values ─────────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT = "10"

# ═══════════════════════════════════════════════════════════════════════════════
# TreeManager
# ═══════════════════════════════════════════════════════════════════════════════


class TreeManager:
    """SQLite-backed tree of memories with LLM-powered branch classification.

    Every memory is placed in a fixed five-level hierarchy:
        root → branch → date → period → memory

    Time layers (date, period) are auto-created by ``add_node()`` —
    the LLM only classifies the root and branch.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file (shared with other stores).
    api_base:
        Base URL for the LLM API (e.g. ``https://api.deepseek.com/v1``).
    api_key:
        API key for authentication.
    model:
        LLM model name (e.g. ``deepseek-v4-flash``).
    """

    def __init__(self, db_path: str, api_base: str, api_key: str, model: str,
                 classifier=None) -> None:
        self._db_path = db_path
        self._api_base = api_base
        self._api_key = api_key
        self._model = model

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_TREE_SQL)
        self._conn.commit()

        if classifier is None:
            from memory_skill.tree_classifier import TreeClassifier
            classifier = TreeClassifier(api_base, api_key, model)
        self._classifier = classifier

        # Ensure root nodes and base branches exist
        self._ensure_roots()

    # ── Initialisation ─────────────────────────────────────────────────────

    def _ensure_roots(self) -> None:
        """Insert root nodes and base branches if they don't exist already."""
        all_nodes = _ROOT_NODES + _BASE_BRANCHES
        for node in all_nodes:
            self._conn.execute(
                "INSERT OR IGNORE INTO tree_nodes (id, parent_id, root, label, level) "
                "VALUES (?, ?, ?, ?, ?)",
                (node["id"], node["parent_id"], node["root"], node["label"], node["level"]),
            )
        self._conn.commit()

    # ── Tree representation ────────────────────────────────────────────────

    def get_tree_json(self) -> str:
        """Return the current tree as an indented text representation for LLM prompts.

        Format::

            user_root (用户记忆根)
              ├── user_pref (偏好)
                ├── user_pref_2026-07-19 (2026-07-19)
                  ├── user_pref_2026-07-19_am (上午)
                    ├── user_pref_2026-07-19_am_a1b2c3d4 (中午吃了炸酱面)
        """
        rows = self._conn.execute(
            "SELECT id, parent_id, root, label, level FROM tree_nodes ORDER BY level, id"
        ).fetchall()

        # Build parent → children map
        children: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            pid = row["parent_id"] or "__root__"
            if pid not in children:
                children[pid] = []
            children[pid].append(row)

        lines: list[str] = []
        for root_node in _ROOT_NODES:
            self._render_subtree(root_node["id"], children, lines, indent=0)

        return "\n".join(lines)

    def _render_subtree(
        self, node_id: str, children: dict[str, list[sqlite3.Row]],
        lines: list[str], indent: int,
    ) -> None:
        """Recursively render a subtree into *lines*."""
        row = self._conn.execute(
            "SELECT id, label FROM tree_nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if not row:
            return

        prefix = "  " * indent
        connector = "├── " if indent > 0 else ""
        lines.append(f"{prefix}{connector}{row['id']} ({row['label']})")

        kids = children.get(node_id, [])
        for child in kids:
            self._render_subtree(child["id"], children, lines, indent + 1)

    def get_tree_text(self) -> str:
        """Pretty-print the tree for debugging (alias for get_tree_json)."""
        return self.get_tree_json()

    # ── Branch Query API ───────────────────────────────────────────────────

    def branch_counts(self, branch_id: str) -> tuple[int, int]:
        """Return (memory_count, date_count) for a branch.

        Safe for external callers — uses the public SQLite connection.
        """
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM tree_nodes WHERE id LIKE ?",
            (f"{branch_id}_%____%",),
        )
        mem = cur.fetchone()[0]
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM tree_nodes WHERE id LIKE ?",
            (f"{branch_id}_20%",),
        )
        dates = cur.fetchone()[0]
        return mem, dates

    def branch_avg_weight(self, branch_id: str) -> float:
        """Average weight of memory entries in a branch."""
        cur = self._conn.execute(
            "SELECT AVG(weight) FROM tree_nodes WHERE id LIKE ?",
            (f"{branch_id}_%",),
        )
        row = cur.fetchone()
        return round(row[0], 3) if row and row[0] is not None else 0.5

    def branch_last_updated(self, branch_id: str) -> str | None:
        """Most recent created_at timestamp in the branch (ISO format string)."""
        cur = self._conn.execute(
            "SELECT MAX(created_at) FROM tree_nodes WHERE id LIKE ?",
            (f"{branch_id}_%",),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] else None

    # ── Skill tree ───────────────────────────────────────────────────────────

    def add_skill_node(self, path: list[str], entry_id: str = "", weight: float = 0.5) -> str:
        """Create a skill-tree node with concept hierarchy from *path*.

        Example: ``add_skill_node(["Python","FastAPI"], "entry_abc")`` creates
        ``assistant_skill/Python/FastAPI/entry_abc``, auto-creating intermediate
        nodes that don't exist yet.
        """
        parent_id = "assistant_skill"
        for i, name in enumerate(path):
            is_leaf = (i == len(path) - 1)
            node_id = f"assistant_skill_{'_'.join(path[:i+1])}"
            cur = self._conn.execute(
                "SELECT 1 FROM tree_nodes WHERE id=?", (node_id,)
            )
            if not cur.fetchone():
                cols = "id,parent_id,root,label,level"
                vals = "?, ?, 'assistant', ?, ?"
                params: list = [node_id, parent_id, name, 2 + i]
                if is_leaf and entry_id:
                    cols += ", memory_ref, weight"
                    vals += ", ?, ?"
                    params += [entry_id, weight]
                self._conn.execute(
                    f"INSERT INTO tree_nodes ({cols}) VALUES ({vals})",
                    params,
                )
            parent_id = node_id
        self._conn.commit()
        return parent_id

    def get_skill_tree_summary(self) -> str:
        """Return indented text of the skill tree for LLM prompts."""
        rows = self._conn.execute(
            "SELECT id, parent_id, label, level FROM tree_nodes"
            " WHERE id LIKE 'assistant_skill%' ORDER BY id"
        ).fetchall()
        if not rows:
            return "(空)"
        lines = []
        for r in rows:
            indent = "  " * max(0, r["level"] - 2)
            lines.append(f"{indent}{r['label']}")
        return "\n".join(lines)

    # ── Classification ─────────────────────────────────────────────────────

    def classify(self, content: str) -> dict:
        return self._classifier.classify(content)

    # ── Node management ────────────────────────────────────────────────────

    def add_node(
        self,
        content: str,
        memory_id: str,
        root: str,
        branch: str,
        timestamp: datetime | None = None,
    ) -> str:
        """Insert a new memory node with auto-created time layers.

        Automatically creates (using INSERT OR IGNORE):
          1. Date node: ``{root}_{branch}_{YYYY-MM-DD}`` under the branch
          2. Period node: ``{root}_{branch}_{YYYY-MM-DD}_{period_key}`` under date
          3. Memory node with truncated content label

        Parameters
        ----------
        content:
            The memory text content.
        memory_id:
            Unique reference ID for this memory (from LearnedStore).
        root:
            One of ``"user"`` or ``"assistant"``.
        branch:
            One of ``"pref"``, ``"mem"``, ``"pers"``, ``"task"``.
        timestamp:
            When the memory was created.  Defaults to ``datetime.now()``.

        Returns
        -------
        str
            The newly created memory node ID.
        """
        if root not in ("user", "assistant"):
            root = "user"
        if branch not in ("pref", "mem", "pers", "task", "skill"):
            branch = "mem"

        ts = timestamp or datetime.now()
        date_str = ts.strftime("%Y-%m-%d")
        hour = ts.hour
        period_key = _period_for_hour(hour)
        period_label = _PERIOD_MAP[period_key]

        branch_id = f"{root}_{branch}"

        # 1. Ensure branch exists (INSERT OR IGNORE)
        self._conn.execute(
            "INSERT OR IGNORE INTO tree_nodes (id, parent_id, root, label, level) "
            "VALUES (?, ?, ?, ?, ?)",
            (branch_id, f"{root}_root", root, branch, 1),
        )

        # 2. Ensure date node exists under branch
        date_id = f"{root}_{branch}_{date_str}"
        self._conn.execute(
            "INSERT OR IGNORE INTO tree_nodes (id, parent_id, root, label, level) "
            "VALUES (?, ?, ?, ?, ?)",
            (date_id, branch_id, root, date_str, 2),
        )

        # 3. Ensure period node exists under date
        period_id = f"{root}_{branch}_{date_str}_{period_key}"
        self._conn.execute(
            "INSERT OR IGNORE INTO tree_nodes (id, parent_id, root, label, level) "
            "VALUES (?, ?, ?, ?, ?)",
            (period_id, date_id, root, period_label, 3),
        )

        # 4. Insert memory node under period
        node_id = f"{period_id}_{uuid.uuid4().hex[:8]}"
        safe_label = content.strip()[:20] if content else "未分类"

        self._conn.execute(
            "INSERT INTO tree_nodes (id, parent_id, root, label, level, memory_ref) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (node_id, period_id, root, safe_label, 4, memory_id),
        )
        self._conn.commit()

        _logger.debug(
            "Tree add_node: %s → branch=%s date=%s period=%s",
            node_id, branch_id, date_str, period_label,
        )
        return node_id

    # ── Context retrieval ──────────────────────────────────────────────────

    def get_context(self, query: str) -> str:
        """Return tree context relevant to *query* for agent prompt injection.

        Supports two query modes:

        1. **Time-based queries** — when *query* contains Chinese time
           keywords ("上午", "下午", "晚上", "今天", "昨天", etc.), returns
           memories from matching period nodes.

        2. **Keyword queries** — searches memory labels for matching text,
           returning the containing branch/date/period context.

        Falls back to a compact tree summary if nothing matches.
        """
        if not query or not query.strip():
            return self._tree_summary()

        # Try time-based query first
        time_result = self._time_context(query)
        if time_result:
            return time_result

        # Fall back to keyword search
        return self._keyword_context(query)

    def _time_context(self, query: str) -> str:
        """Resolve time-based queries like "今天上午发生了什么"."""
        from datetime import date, timedelta

        today = date.today()
        target_date = today

        # Detect date offset
        if "昨天" in query:
            target_date = today - timedelta(days=1)
        elif "前天" in query:
            target_date = today - timedelta(days=2)
        elif "明天" in query:
            target_date = today + timedelta(days=1)
        # "今天" is default (today)

        date_str = target_date.strftime("%Y-%m-%d")

        # Detect period
        periods_to_search: list[str] = []
        if "早上" in query or "上午" in query or "早晨" in query:
            periods_to_search = ["am"]
        elif "中午" in query or "下午" in query:
            periods_to_search = ["pm"]
        elif "晚上" in query or "夜里" in query or "夜间" in query:
            periods_to_search = ["ev"]

        # If no time keywords detected at all, return empty (not a time query)
        if not periods_to_search:
            return ""

        # Build period IDs to search across all branches
        context_lines: list[str] = [f"[记忆树时间检索 — {target_date.strftime('%Y-%m-%d')}]"]
        found_any = False

        for branch_node in _BASE_BRANCHES:
            branch_id = branch_node["id"]
            for pk in periods_to_search:
                period_id = f"{branch_id}_{date_str}_{pk}"
                # Verify period node exists
                period_row = self._conn.execute(
                    "SELECT id FROM tree_nodes WHERE id = ?", (period_id,)
                ).fetchone()
                if not period_row:
                    continue

                # Fetch memories under this period
                mems = self._conn.execute(
                    "SELECT id, label, memory_ref FROM tree_nodes "
                    "WHERE parent_id = ? AND level = 4 ORDER BY created_at LIMIT 10",
                    (period_id,),
                ).fetchall()

                if mems:
                    found_any = True
                    period_label = _PERIOD_MAP[pk]
                    context_lines.append(
                        f"  [{branch_node['label']}] → {period_label}:"
                    )
                    for m in mems:
                        context_lines.append(f"    • {m['label']}")

        if not found_any:
            return ""  # No hits — fall through to keyword search

        return "\n".join(context_lines)

    def _keyword_context(self, query: str) -> str:
        """Search tree nodes by label keyword match."""
        terms = query.strip().split()
        matched_ids: set[str] = set()
        matched_rows: list[sqlite3.Row] = []

        for term in terms:
            if len(term) < 2:
                continue
            rows = self._conn.execute(
                "SELECT id, parent_id, root, label, level, memory_ref "
                "FROM tree_nodes WHERE label LIKE ? "
                "LIMIT 5",
                (f"%{term}%",),
            ).fetchall()
            for row in rows:
                if row["id"] not in matched_ids:
                    matched_ids.add(row["id"])
                    matched_rows.append(row)

        if not matched_rows:
            return self._tree_summary()

        context_lines: list[str] = ["[记忆树相关分支]"]
        seen: set[str] = set()

        for row in matched_rows[:5]:
            self._render_context_chain(row, context_lines, seen)

        if len(context_lines) <= 1:
            return self._tree_summary()

        return "\n".join(context_lines)

    def _render_context_chain(
        self, mem_row: sqlite3.Row, lines: list[str], seen: set[str],
    ) -> None:
        """Walk up the chain: memory → period → date → branch → root."""
        chain: list[tuple[str | None, str]] = []  # (parent_id, label)
        current_id: str | None = mem_row["parent_id"]
        depth = 0
        max_depth = 5  # safety

        while current_id and depth < max_depth:
            parent = self._conn.execute(
                "SELECT parent_id, label FROM tree_nodes WHERE id = ?",
                (current_id,),
            ).fetchone()
            if not parent:
                break
            if current_id not in seen:
                seen.add(current_id)
                chain.append((parent["parent_id"], parent["label"]))
            current_id = parent["parent_id"]
            depth += 1

        if not chain:
            return

        # Build breadcrumb: root → branch → date → period → memory
        breadcrumb = " → ".join(lbl for _, lbl in reversed(chain))
        lines.append(f"  {breadcrumb} → {mem_row['label']}")

        # Show siblings (other memories in same period)
        if mem_row["parent_id"]:
            sibs = self._conn.execute(
                "SELECT label FROM tree_nodes "
                "WHERE parent_id = ? AND id != ? LIMIT 5",
                (mem_row["parent_id"], mem_row["id"]),
            ).fetchall()
            for sib in sibs:
                lines.append(f"    • {sib['label']}")

    def _tree_summary(self) -> str:
        """Return a compact tree summary for when no specific match is found."""
        lines: list[str] = ["[记忆树概览]"]
        for branch in _BASE_BRANCHES:
            branch_id = branch["id"]
            # Count memory nodes under this branch
            mem_count = self._conn.execute(
                "SELECT COUNT(*) FROM tree_nodes WHERE id LIKE ? || '_%' AND level = 4",
                (branch_id,),
            ).fetchone()[0]
            # Skill branch uses skill/sub-skill hierarchy, not date layers.
            if branch_id == "assistant_skill":
                skill_count = self._conn.execute(
                    "SELECT COUNT(*) FROM tree_nodes WHERE parent_id = ? AND level = 2",
                    (branch_id,),
                ).fetchone()[0]
                lines.append(f"  {branch['label']}: {skill_count} 个技能")
            else:
                date_count = self._conn.execute(
                    "SELECT COUNT(*) FROM tree_nodes WHERE parent_id = ? AND level = 2",
                    (branch_id,),
                ).fetchone()[0]
                lines.append(
                    f"  {branch['label']}: {mem_count} 条记忆, {date_count} 个日期"
                )
        return "\n".join(lines)

    # ── LLM-navigated tree context ─────────────────────────────────────────

    def navigate(self, query: str) -> str:
        """LLM-navigated tree context. Returns formatted memory context
        from LLM-selected branches, with fallback to all branches.

        On LLM failure, falls back to last day across all 4 branches.
        """
        searches = self._llm_navigate(query)
        if not searches:
            searches = self._fallback_navigate()
        return self._resolve_searches(searches)

    def navigate_without_llm(self) -> str:
        """Deterministic tree context with NO LLM involvement.

        Used by the passive weave path (ADR-0002: weave is pure context
        assembly, no internal LLM). Agents that want LLM-selected branches
        call ``navigate(query)`` explicitly through a tool instead.
        """
        return self._resolve_searches(self._fallback_navigate())

    def _llm_navigate(self, query: str) -> list[dict] | None:
        """Ask the classifier (LLM) to select branches + time ranges.

        Returns a list of ``{"branch": str, "days": int}`` dicts,
        or ``None`` on any failure.
        """
        return self._classifier.navigate(query)

    def _fallback_navigate(self) -> list[dict]:
        """Return all 5 branches × last 1 day when LLM is unavailable."""
        return [
            {"branch": "user_pref", "days": 1},
            {"branch": "user_mem", "days": 1},
            {"branch": "assistant_pers", "days": 1},
            {"branch": "assistant_task", "days": 1},
            {"branch": "assistant_skill", "days": 1},
        ]

    def _resolve_searches(self, searches: list[dict]) -> str:
        """Resolve ``{branch, days}`` searches to actual tree nodes.

        Queries memory nodes (level=4) under matching date+period nodes
        and returns formatted context text.
        """
        from datetime import date, timedelta

        today = date.today()
        blocks: list[str] = []

        for s in searches:
            branch_id = s["branch"]
            days = max(1, min(s["days"], 30))
            branch_label = _BRANCH_LABEL_MAP.get(branch_id, branch_id)

            if days == 1:
                header = f"[树导航: {branch_label}, 最近1天]"
            else:
                header = f"[树导航: {branch_label}, 最近{days}天]"

            branch_lines: list[str] = []
            for offset in range(days):
                target_date = today - timedelta(days=offset)
                date_str = target_date.strftime("%Y-%m-%d")

                for pk, plabel in _PERIOD_MAP.items():
                    period_id = f"{branch_id}_{date_str}_{pk}"
                    mems = self._conn.execute(
                        "SELECT label, created_at FROM tree_nodes "
                        "WHERE parent_id = ? AND level = 4 "
                        "ORDER BY created_at DESC LIMIT 10",
                        (period_id,),
                    ).fetchall()
                    if mems:
                        for m in mems:
                            mem_time = m["created_at"]
                            short_time = _format_tree_time(mem_time)
                            branch_lines.append(
                                f"[{short_time} {plabel}] {m['label']}"
                            )

            if branch_lines:
                blocks.append(header)
                blocks.extend(branch_lines)

        if not blocks:
            return "[树导航] 暂无相关记忆"

        return "\n".join(blocks)


def _format_tree_time(ts: str) -> str:
    """Format a datetime string like '2026-07-19 14:30:00' → '07-19'."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%m-%d")
    except (ValueError, TypeError):
        return ts[:10] if ts else "??-??"

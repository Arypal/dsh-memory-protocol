#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从本地记忆库 / 会话缓存中清除误发的明文凭据（sk-...）。

默认 dry-run：只报告命中位置与条数，不改任何东西。
加 --apply 才真正删除/脱敏：
  - memory.db 各表：把含凭据的值替换成 [REDACTED]，并重建 dialogue_fts
  - memory.db_chroma：删除对应向量
  - storages/session_projcache 缓存文件与 tool_calls.log：明文替换成 [REDACTED]

2026-10-28 重建：原文件在 .dsh 迁移中损坏，按原行为（.pyc 反查）重写。
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys

from _common import DB, CHROMA, MEMORY_DIR, dsh_home  # noqa: E402

SESSION_ROOT = str(dsh_home() / "storages" / "session_projcache")
TOOL_LOG = str(MEMORY_DIR / "tool_calls.log")
PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")

TARGETS = [
    ("dialogue_turns", "content"),
    ("facts", "current_value"),
    ("fact_evidence", "quote"),
    ("pending_candidates", "summary"),
    ("pending_candidates", "evidence_json"),
    ("episode_summaries", "summary"),
]


def redact(text: str) -> str:
    return PATTERN.sub("[REDACTED]", text)


def scan_sqlite(apply: bool) -> int:
    if not os.path.isfile(DB):
        print("memory.db not found:", DB)
        return 0
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    hits = 0
    for table, col in TARGETS:
        try:
            cur.execute("SELECT rowid, %s FROM %s WHERE %s LIKE '%%sk-%%'" % (col, table, col))
            rows = cur.fetchall()
        except sqlite3.Error:
            continue
        for rowid, value in rows:
            if value and PATTERN.search(str(value)):
                hits += 1
                preview = str(value)[:60].replace("\n", " ")
                print("[sqlite] %s.%s rowid=%s: %s..." % (table, col, rowid, preview))
                if apply:
                    cur.execute(
                        "UPDATE %s SET %s=? WHERE rowid=?" % (table, col),
                        (redact(str(value)), rowid),
                    )
    if apply and hits:
        try:
            cur.execute("INSERT INTO dialogue_fts(dialogue_fts) VALUES('rebuild')")
        except sqlite3.Error:
            pass
        conn.commit()
    conn.close()
    print("[sqlite] hits=%d%s" % (hits, " (redacted)" if apply else " (dry-run)"))
    return hits


def scan_chroma(apply: bool) -> int:
    if not os.path.isdir(CHROMA):
        print("chroma dir not found:", CHROMA)
        return 0
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA)
        total = 0
        for coll in client.list_collections():
            name = coll if isinstance(coll, str) else getattr(coll, "name", str(coll))
            try:
                collection = client.get_collection(name)
                got = collection.get(include=["documents"])
            except Exception:  # noqa: BLE001
                continue
            ids = got.get("ids") or []
            docs = got.get("documents") or []
            bad = []
            for i, doc in enumerate(docs):
                if doc and PATTERN.search(str(doc)):
                    bad.append(ids[i])
                    print("[chroma] %s id=%s: %s..." % (name, ids[i], str(doc)[:60].replace("\n", " ")))
            total += len(bad)
            if apply and bad:
                try:
                    collection.delete(ids=bad)
                except Exception:  # noqa: BLE001
                    pass
        print("[chroma] hits=%d%s" % (total, " (deleted)" if apply else " (dry-run)"))
        return total
    except Exception as exc:  # noqa: BLE001
        print("[chroma] skipped:", exc)
        return 0


def scan_sessions(apply: bool) -> int:
    hits = 0
    if os.path.isdir(SESSION_ROOT):
        for dirpath, _dirnames, filenames in os.walk(SESSION_ROOT):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:  # noqa: BLE001
                    continue
                if PATTERN.search(text):
                    hits += 1
                    print("[session] %s" % path)
                    if apply:
                        try:
                            with open(path, "w", encoding="utf-8") as f:
                                f.write(redact(text))
                        except Exception:  # noqa: BLE001
                            pass
    if os.path.isfile(TOOL_LOG):
        try:
            with open(TOOL_LOG, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if PATTERN.search(text):
                hits += 1
                print("[log] %s" % TOOL_LOG)
                if apply:
                    with open(TOOL_LOG, "w", encoding="utf-8") as f:
                        f.write(redact(text))
        except Exception:  # noqa: BLE001
            pass
    print("[session] hits=%d%s" % (hits, " (redacted)" if apply else " (dry-run)"))
    return hits


def main() -> int:
    apply = "--apply" in sys.argv[1:]
    print("=== purge secrets (apply=%s) ===" % apply)
    total = 0
    total += scan_sqlite(apply)
    total += scan_chroma(apply)
    total += scan_sessions(apply)
    if total == 0:
        print("clean: 没发现 sk- 明文")
    elif not apply:
        print("（dry-run：加 --apply 才会真删）")
    else:
        print("done: cleaned %d location(s)" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())

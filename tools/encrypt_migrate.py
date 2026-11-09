#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""历史数据加密迁移（幂等，先备份再改）。

把库里还在明文的 facts / fact_history / fact_evidence / dialogue_turns
以及 chroma learned documents 全部转成 `enc:v1:` 密文。已经是密文的跳过。

用法：
    python3 encrypt_migrate.py --dry-run   # 只看会迁移多少条
    python3 encrypt_migrate.py --apply     # 先备份，再迁移

注意：dialogue_fts 索引保持明文 token（可检索性 vs 加密的取舍，见 AGENTS.md）。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

from _common import DB, MEMORY_DIR, PLUGIN_PY, TOOLS_DIR  # noqa: E402

DB = str(DB)
MEMORY_DIR = str(MEMORY_DIR)
sys.path.insert(0, str(PLUGIN_PY))
sys.path.insert(0, str(TOOLS_DIR))
# 必须显式覆盖：从 DSH 进程继承到的 MEMORY_SKILL_DB_PATH 可能指向 C 盘旧库，
# setdefault 会保留它 → secure 用旧库目录的 master.key 加密，服务端解不开。
os.environ["MEMORY_SKILL_DB_PATH"] = DB
from memory_skill import secure  # noqa: E402

_PREFIX = "enc:v1:"


def migrate_sqlite(apply: bool) -> dict:
    stats: dict[str, int] = {}
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA busy_timeout=20000")
    try:
        for table, pk, col in (
            ("facts", "id", "current_value"),
            ("fact_history", "id", "value"),
            ("fact_evidence", "id", "quote"),
            ("dialogue_turns", "id", "content"),
        ):
            rows = con.execute(
                f"SELECT {pk}, {col} FROM {table} WHERE {col} NOT LIKE ?",
                (_PREFIX + "%",),
            ).fetchall()
            stats[table] = len(rows)
            if apply and rows:
                con.executemany(
                    f"UPDATE {table} SET {col}=? WHERE {pk}=?",
                    [(secure.encrypt(v), i) for i, v in rows],
                )
                con.commit()
    finally:
        con.close()
    return stats


def migrate_learned(apply: bool) -> dict:
    import chromadb
    cli = chromadb.PersistentClient(path=DB + "_chroma")
    col = cli.get_collection("learned_default")
    # 必须连 embeddings 一起取回：update 不传 embeddings 时 Chroma 会用它的
    # 默认 embedding function 重新嵌入（会触发下载 all-MiniLM 模型）。
    data = col.get(include=["documents", "embeddings"])
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    embs = data.get("embeddings")
    if embs is None:
        embs = [None] * len(ids)
    todo = [(i, d, e) for i, d, e in zip(ids, docs, embs)
            if d and not str(d).startswith(_PREFIX)]
    if apply and todo:
        for start in range(0, len(todo), 200):
            chunk = todo[start:start + 200]
            kwargs = {
                "ids": [i for i, _, _ in chunk],
                "documents": [secure.encrypt(d) for _, d, _ in chunk],
            }
            emb_list = []
            for _, _, e in chunk:
                if e is None:
                    emb_list = []
                    break
                emb_list.append(e.tolist() if hasattr(e, "tolist") else list(e))
            if emb_list:
                kwargs["embeddings"] = emb_list
            col.update(**kwargs)
    return {"learned_documents": len(todo)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    apply = bool(args.apply)
    if apply:
        import memory_backup as mb
        print("backup:", json.dumps(mb.backup(), ensure_ascii=False))
    out = {"mode": "apply" if apply else "dry-run"}
    out.update(migrate_sqlite(apply))
    out.update(migrate_learned(apply))
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

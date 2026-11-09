#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修密钥错配：把用 C 盘 master.key 加密的数据，重新用 G 盘正本 key 加密。

背景：加密迁移脚本继承到了旧环境变量 MEMORY_SKILL_DB_PATH（指向 C 盘旧库），
用了 C 盘的 master.key 加密；服务端 proxy 用 G 盘正本 key 解密 → 全部损坏。

用法：
    python3 rekey.py --dry-run
    python3 rekey.py --apply
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import sqlite3
import sys

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from _common import DB, MASTER_KEY  # noqa: E402

KEY_G = str(MASTER_KEY)
KEY_C = os.environ.get("OLD_MASTER_KEY_PATH", "")
PREFIX = "enc:v1:"


def load_key(path: str) -> bytes:
    with open(path, "rb") as f:
        return base64.b64decode(f.read().strip())


KG = load_key(KEY_G)
KC = load_key(KEY_C) if KEY_C and os.path.exists(KEY_C) else None
if KC is None:
    print("[rekey] 未提供 OLD_MASTER_KEY_PATH，含旧 key 的数据将记为 fail", file=sys.stderr)


def dec(key: bytes, text: str) -> str:
    raw = base64.b64decode(text[len(PREFIX):])
    nonce, ct = raw[:12], raw[12:]
    return AESGCM(key).decrypt(nonce, ct, None).decode("utf-8")


def enc(key: bytes, text: str) -> str:
    nonce = secrets.token_bytes(12)
    ct = AESGCM(key).encrypt(nonce, text.encode("utf-8"), None)
    return PREFIX + base64.b64encode(nonce + ct).decode("ascii")


def classify(value) -> tuple[str, str]:
    """返回 (new_value, action)，action ∈ keep/rekey/fail/plain。"""
    if not value or not str(value).startswith(PREFIX):
        return value, "plain"
    try:
        dec(KG, value)
        return value, "keep"
    except Exception:
        pass
    try:
        return enc(KG, dec(KC, value)), "rekey"
    except Exception:
        return value, "fail"


def sqlite_rekey(apply: bool) -> dict:
    stats = {"keep": 0, "rekey": 0, "fail": 0, "plain": 0}
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA busy_timeout=20000")
    try:
        for table, pk, col in (
            ("facts", "id", "current_value"),
            ("fact_history", "id", "value"),
            ("fact_evidence", "id", "quote"),
            ("dialogue_turns", "id", "content"),
        ):
            updates = []
            for i, v in con.execute(f"SELECT {pk}, {col} FROM {table}").fetchall():
                nv, act = classify(v)
                stats[act] += 1
                if act == "rekey" and apply:
                    updates.append((nv, i))
            if apply and updates:
                con.executemany(
                    f"UPDATE {table} SET {col}=? WHERE {pk}=?", updates)
                con.commit()
    finally:
        con.close()
    return stats


def chroma_rekey(apply: bool) -> dict:
    import chromadb
    cli = chromadb.PersistentClient(path=DB + "_chroma")
    col = cli.get_collection("learned_default")
    data = col.get(include=["documents", "embeddings"])
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    embs = data.get("embeddings")
    if embs is None or len(embs) == 0:
        embs = [None] * len(ids)
    stats = {"keep": 0, "rekey": 0, "fail": 0, "plain": 0, "no_embedding": 0}
    todo = []
    for i, d, e in zip(ids, docs, embs):
        nv, act = classify(d)
        stats[act] += 1
        if act == "rekey" and apply:
            if e is None:
                stats["no_embedding"] += 1
                continue
            todo.append((i, nv, e))
    for start in range(0, len(todo), 200):
        chunk = todo[start:start + 200]
        col.update(
            ids=[i for i, _, _ in chunk],
            documents=[d for _, d, _ in chunk],
            embeddings=[e.tolist() if hasattr(e, "tolist") else list(e)
                        for _, _, e in chunk],
        )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    out = {"mode": "apply" if args.apply else "dry-run"}
    out.update(sqlite_rekey(args.apply))
    out.update({f"chroma_{k}": v for k, v in chroma_rekey(args.apply).items()})
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

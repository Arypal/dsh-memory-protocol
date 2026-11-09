#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""记忆库备份金字塔（加密、写入即只读）。

用法：
    python3 memory_backup.py              # 今天的日备份（幂等）
    python3 memory_backup.py --list       # 列出所有备份
    python3 memory_backup.py --restore <file>   # 恢复（会先做一次安全备份）

布局：<memory_dir>/backups/{daily,weekly,monthly,yearly}/<date>.mbk
  - daily  保留 7 份
  - weekly 每周一产一份，保留 4 份
  - monthly 每月 1 日一份，保留 12 份
  - yearly  每年 1 月 1 日一份，保留 5 份
  - 文件写入后设只读（不可变）；内容用 master.key 做 AES-256-GCM 加密，
    可脱离程序用同一把 key 手工恢复。

诚实边界：备份是用**同一把 master.key** 加密的——密钥和备份放在同一台机器，
挡得住备份被随手外带/上传到网盘，挡不住能拿到整台机器的人。
"""
from __future__ import annotations

import argparse
import base64
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import date, datetime

from _common import MEMORY_DIR, DB, CHROMA, BACKUP_ROOT, PLUGIN_PY  # noqa: E402
KEEP = {"daily": 7, "weekly": 4, "monthly": 12, "yearly": 5}


def _key() -> bytes:
    sys.path.insert(0, str(PLUGIN_PY))
    from memory_skill import secure
    return secure._load_or_create_key()


def _encrypt_bytes(data: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = os.urandom(12)
    return b"MEMBACKUP1" + nonce + AESGCM(_key()).encrypt(nonce, data, None)


def _decrypt_bytes(blob: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if not blob.startswith(b"MEMBACKUP1"):
        raise ValueError("not a memory backup file")
    nonce, ct = blob[10:22], blob[22:]
    return AESGCM(_key()).decrypt(nonce, ct, None)


def _readonly(path: str) -> None:
    try:
        os.chmod(path, 0o444)
        import stat
        os.chflags  # noqa: B018 - not available on Windows
    except Exception:
        pass
    try:
        import ctypes
        ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x01)  # READONLY
    except Exception:
        pass


def _write_backup(target_dir: str, day: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    out = os.path.join(target_dir, f"{day}.mbk")
    if os.path.exists(out):
        return out  # 幂等
    tmp = tempfile.mkdtemp(prefix="membackup-")
    try:
        zip_path = os.path.join(tmp, "payload.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            if os.path.exists(DB):
                z.write(DB, "memory.db")
            if os.path.isdir(CHROMA):
                for root, _dirs, files in os.walk(CHROMA):
                    for fn in files:
                        p = os.path.join(root, fn)
                        z.write(p, os.path.join("memory.db_chroma",
                                                os.path.relpath(p, CHROMA)))
        blob = _encrypt_bytes(open(zip_path, "rb").read())
        with open(out, "wb") as f:
            f.write(blob)
        _readonly(out)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _cleanup(kind: str) -> int:
    d = os.path.join(BACKUP_ROOT, kind)
    if not os.path.isdir(d):
        return 0
    files = sorted(f for f in os.listdir(d) if f.endswith(".mbk"))
    removed = 0
    for f in files[:-KEEP[kind]] if len(files) > KEEP[kind] else []:
        try:
            os.chmod(os.path.join(d, f), 0o666)
            os.remove(os.path.join(d, f))
            removed += 1
        except Exception:
            pass
    return removed


def backup() -> dict:
    today = date.today()
    made = []
    out = _write_backup(os.path.join(BACKUP_ROOT, "daily"), today.isoformat())
    made.append(out)
    if today.weekday() == 0:  # 周一
        made.append(_write_backup(os.path.join(BACKUP_ROOT, "weekly"),
                                  today.isoformat()))
    if today.day == 1:
        made.append(_write_backup(os.path.join(BACKUP_ROOT, "monthly"),
                                  today.isoformat()))
        if today.month == 1:
            made.append(_write_backup(os.path.join(BACKUP_ROOT, "yearly"),
                                      today.isoformat()))
    purged = {k: _cleanup(k) for k in KEEP}
    return {"status": "ok", "files": made, "purged": purged}


def list_backups() -> list[dict]:
    out = []
    for kind in KEEP:
        d = os.path.join(BACKUP_ROOT, kind)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".mbk"):
                continue
            p = os.path.join(d, f)
            out.append({"kind": kind, "name": f,
                        "size": os.path.getsize(p),
                        "mtime": datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")})
    return out


def restore(path: str) -> dict:
    if not os.path.exists(path):
        return {"status": "error", "reason": "backup not found"}
    data = _decrypt_bytes(open(path, "rb").read())
    tmp = tempfile.mkdtemp(prefix="memrestore-")
    try:
        zip_path = os.path.join(tmp, "payload.zip")
        with open(zip_path, "wb") as f:
            f.write(data)
        safe = os.path.join(tmp, f"safety-{datetime.now():%Y%m%d-%H%M%S}.mbk")
        if os.path.exists(DB):
            shutil.copyfile(DB, safe)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        src_db = os.path.join(tmp, "memory.db")
        if os.path.exists(src_db):
            shutil.copyfile(src_db, DB)
        src_ch = os.path.join(tmp, "memory.db_chroma")
        if os.path.isdir(src_ch):
            for root, _dirs, files in os.walk(src_ch):
                for fn in files:
                    p = os.path.join(root, fn)
                    rel = os.path.relpath(p, src_ch)
                    dst = os.path.join(CHROMA, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(p, dst)
        return {"status": "ok", "restored": path, "safety_copy": safe}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--restore", metavar="FILE")
    args = ap.parse_args()
    if args.list:
        for b in list_backups():
            print(f"{b['kind']:8} {b['name']:22} {b['size']/1024:8.1f} KB  {b['mtime']}")
        return 0
    if args.restore:
        print(restore(args.restore))
        return 0
    import json
    print(json.dumps(backup(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按需 distill：会话开始时跑一次，自己判断该不该跑。

规则（全部自动，无需参数）：
  - 今天已经跑过        → skip
  - pending 未审 >= 3   → skip（等审核，避免候选堆成山）
  - 否则                → 提炼最近 24 小时（since_days=1, limit=60）

用法：
    python3 G:\\DSH\\_scripts\\daily_distill.py          # 正常检查
    python3 G:\\DSH\\_scripts\\daily_distill.py --force  # 忽略上面两条门槛

日志：G:\\DSH\\_scripts\\distill.log
状态：G:\\DSH\\_scripts\\distill_state.json

2026-10-28 重建：原文件在 .dsh 迁移中损坏，按原逻辑（.pyc 反查）重写。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

from _common import PLUGIN_PY, DB, TOOLS_DIR  # noqa: E402

STATE = str(TOOLS_DIR / "distill_state.json")
LOG = str(TOOLS_DIR / "distill.log")


def log(msg: str) -> None:
    line = "[%s] %s\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:  # noqa: BLE001
        pass


def load_state() -> dict:
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def save_state(data: dict) -> None:
    try:
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        log("state save failed: " + str(exc))


def _open_pending_count() -> int:
    try:
        import sqlite3
        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM pending_candidates WHERE status='open'")
        n = cur.fetchone()[0]
        conn.close()
        return int(n)
    except Exception:  # noqa: BLE001
        return 0


def main() -> int:
    force = "--force" in sys.argv[1:]
    state = load_state()
    today = datetime.now().strftime("%Y-%m-%d")
    last = state.get("last_run", "")

    if not force and last == today:
        log("skip: already ran today (last=%s, open=%d)" % (last, _open_pending_count()))
        return 0

    open_n = _open_pending_count()
    if not force and open_n >= 3:
        log("skip: %d pending candidates awaiting review" % open_n)
        return 0

    sys.path.insert(0, PLUGIN_PY)
    try:
        from memory_skill import MemorySkill, MemorySkillConfig
        from memory_skill.distill import distill

        skill = MemorySkill(MemorySkillConfig(db_path=DB))
        ds = getattr(skill, "dialogue_store", None) or getattr(skill, "dialogue", None)
        ps = getattr(skill, "pending_store", None) or getattr(skill, "pending", None)
        if ds is None or ps is None:
            raise RuntimeError("cannot locate dialogue_store/pending_store on MemorySkill")

        t0 = datetime.now()
        result = distill(ds, ps, since_days=1, limit=60)
        elapsed = (datetime.now() - t0).total_seconds()
        stored = int(result.get("stored", 0) or 0)
        rejected = int(result.get("rejected", 0) or 0)
        note = str(result.get("error") or result.get("note") or "")
        log("distill done in %.1fs: stored=%d rejected=%d note=%s" % (elapsed, stored, rejected, note))
        state["last_run"] = today
        state["last_stored"] = stored
        save_state(state)
        if stored:
            print("%d candidates pending; run memory_pending to review" % stored)
        return 0
    except Exception as exc:  # noqa: BLE001
        log("distill failed: %r" % (exc,))
        print("distill failed: " + repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成只含示例数据的记忆库（不含任何真实记忆）。

用法：
    python samples/make_sample_db.py                # 生成 samples/sample.db
    python samples/make_sample_db.py --db /tmp/x.db
    MEMORY_MODEL_PATH=... python samples/make_sample_db.py

说明：事实卡走 ``MemorySystem.ingest_fact``，对话走 ``MemorySystem.ingest``，
与线上链路一致；没有嵌入模型时检索会降级为 BM25，仍然可用。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

from memory_skill.contracts import DialogueTurn, MemorySkillConfig  # noqa: E402
from memory_skill._compose import _build_system  # noqa: E402

FACTS = [
    ("pref", "primary_language", "示例：主要用 Python 写脚本，需要前端时用 TypeScript。",
     "example fact card — 可随意删除"),
    ("pref", "editor", "示例：编辑器用 VS Code，终端用 Windows Terminal。",
     "example fact card"),
    ("pers", "display_name", "示例：助手自称 Sample，称呼用户为「你」。",
     "example fact card"),
    ("conclusion", "deploy_shape", "示例：单机 stdio 模式部署，不依赖常驻 HTTP 服务。",
     "example fact card"),
]

TURNS = [
    ("user", "帮我把这个目录里的日志按日期归类。"),
    ("assistant", "按 YYYY-MM-DD 建子目录，把同一天的文件移进去；重复运行是幂等的。"),
    ("user", "为什么不用一个常驻服务来处理？"),
    ("assistant", "因为无状态推理 + 外部记忆已经够用；常驻进程只增加运维面。"),
    ("user", "那就把这条结论记下来。"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "samples" / "sample.db"))
    ap.add_argument("--force", action="store_true", help="覆盖已存在的库")
    args = ap.parse_args()

    db = Path(args.db)
    if db.exists():
        if not args.force:
            print(f"[sample] {db} 已存在，加 --force 覆盖")
            return 1
        db.unlink()

    # 双保险：store 层可能读 env，避免误写进真实库
    os.environ["MEMORY_SKILL_DB_PATH"] = str(db)
    cfg = MemorySkillConfig(db_path=str(db))
    system = _build_system(cfg)

    for cat, key, value, quote in FACTS:
        system.ingest_fact(cat, key, value, quote=quote)
    print(f"[sample] 事实卡 {len(FACTS)} 张")

    base = datetime.now() - timedelta(minutes=5 * len(TURNS))
    for i, (role, text) in enumerate(TURNS):
        system.ingest(
            DialogueTurn(
                id=f"sample-{i:03d}",
                role=role,
                content=text,
                timestamp=base + timedelta(minutes=5 * i),
            )
        )
    print(f"[sample] 对话 {len(TURNS)} 轮")
    print(f"[sample] 完成：{db}")
    print("[sample] 用法：MEMORY_SKILL_DB_PATH=%s" % db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布版公共路径解析 —— tools/ 下所有脚本共用。

优先级：显式环境变量 > DSH_HOME > ~/.dsh
  MEMORY_DIR            记忆库目录（默认 <DSH_HOME>/memory）
  MEMORY_SKILL_DB_PATH  单个库文件（默认 <MEMORY_DIR>/memory.db）
"""
from __future__ import annotations

import os
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
PLUGIN_PY = REPO_ROOT / "python"


def dsh_home() -> Path:
    return Path(os.environ.get("DSH_HOME") or (Path.home() / ".dsh"))


def memory_dir() -> Path:
    return Path(os.environ.get("MEMORY_DIR") or (dsh_home() / "memory"))


MEMORY_DIR = memory_dir()
DB = Path(os.environ.get("MEMORY_SKILL_DB_PATH") or (MEMORY_DIR / "memory.db"))
CHROMA = MEMORY_DIR / "memory.db_chroma"
BACKUP_ROOT = MEMORY_DIR / "backups"
MASTER_KEY = MEMORY_DIR / "master.key"

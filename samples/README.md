# samples

`samples/make_sample_db.py` 生成一个**只含示例数据**的记忆库，用于：

- 冒烟测试（`node test/smoke-mcp.mjs`）
- 30 组检索回归（`node tools/test-memory-30.mjs`）
- 上手时先看一张满的库长什么样，再决定要不要清空

```bash
python samples/make_sample_db.py --force
MEMORY_SKILL_DB_PATH=$PWD/samples/sample.db node test/smoke-mcp.mjs
```

生成的 `sample.db` 与 `sample.db_chroma/` 已被 `.gitignore` 忽略，不会误提交。
示例内容全部虚构（4 张事实卡 + 5 轮对话），不含任何真实记忆。

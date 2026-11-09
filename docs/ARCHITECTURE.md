# 架构与数据流

## 1. 进程拓扑

```
DSH 宿主进程
 ├─ lib/index.js（协议插件）
 │   ├─ agent/pre-step        → memory_weave，注入本轮上下文
 │   ├─ agent/turn-stopping   → memory_ingest，user/assistant 分角色入库
 │   └─ tools/pre-execute     → 白名单校验 + 软提醒
 └─ mcp-client(stdio)
     └─ node scripts/run-mcp.cjs
         └─ python -m memory_skill.mcp_server   （PYTHONPATH 指向 python/）
             └─ 21 个工具 → stores
```

## 2. 存储层

| 层 | 模块 | 表 / 后端 |
|---|---|---|
| L1 事实卡 | `fact_store.py` | `facts`、`fact_history`、`fact_evidence`、`entry_aliases`、`scope_bindings` |
| L2 证据 | `fact_store.py` | `fact_evidence`（原话副本，删对话不影响） |
| L3 流水 | `dialogue_store.py` | `dialogue_turns` + `dialogue_fts`（FTS5） |
| L4 摘要 | `retention.py` | `episode_summaries`、`trash_turns` |
| 语义库 | `learned_store.py` + `embedder.py` | ChromaDB collection + ONNX bge |
| 关系 | `tree.py` / `capability_registry.py` | `tree_nodes`、`mem_edges` |
| 学习态 | `learning_queue.py` / `pending_store.py` | `learning_queue`、`pending_candidates` |
| 短期观测 | `saw_buffer.py` | 内存环形缓冲（无采集源时为空转） |

## 3. 一轮对话的完整链路

```
用户消息
  │
  ├─ pre-step: memory_weave(query)
  │    ├─ tier1  场景感知：近期状态 / 教训 / 进行中
  │    ├─ tier2  深记忆：事实卡优先 + 三腿 RRF 召回
  │    ├─ nudge   高权重提醒（锚点 / 记忆树导航）
  │    ├─ 预算   1800 字符上限 + 优先级裁剪
  │    └─ gate   [协议提醒]（软提醒，不阻断）
  │
  ├─ 推理（模型按需调工具：search / fact_add / teach_skill …）
  │
  └─ turn-stopping: memory_ingest
       ├─ user/message      → dialogue_turns + FTS + 向量
       ├─ assistant/message → 同上（分角色，不混淆证据）
       └─ 观察固化           → observation.py（无 LLM）
```

## 4. 检索管线（retriever.py）

1. **候选生成**：向量 top-k、BM25 top-k、别名精确命中、时间窗过滤。
2. **融合**：RRF（`rrf_k=60`），腿权重可配——别名 3.5 > BM25 2.5 > 语义 0.5 ≈ 时间 0.5；
   技能腿在检索侧单独加权 2.0。
3. **硬优先级**：事实卡被别名强命中时压过历史问句配对答案；未命中时不启用问句降权（避免误伤）。
4. **元记忆隔离**：`lesson` / `topic` / `context` 不进普通检索，只走注入通道。
5. **去重**：同 id 合并只提权，绝不改写已存文本。

## 5. 主动学习闭环

```
memory_classify(category, gaps)
   └→ learning_queue（skill 缺口 / mission）
        └→ memory_distill（窗口 60 轮，LLM 打分）
             └→ pending_candidates
                  ├─ reject → 丢弃
                  └─ accept → 事实卡 / 技能（teach_skill）
```

## 6. 加密与备份

- `secure.py`：字段级 AES-256-GCM，密文 `enc:v1:<base64(nonce|ct)>`，密钥 `<MEMORY_DIR>/master.key`。
- 解密失败不静默：返回 `[! 通不过完整性校验]`，`memory_status.integrity` 报告条数。
- `tools/memory_backup.py`：整库 + chroma 打包、AES 加密、写入即只读，四档保留策略。

## 7. 设计取舍（踩过的坑）

| 取舍 | 原因 |
|---|---|
| 密钥与数据同目录 | 插件必须在无人值守时解密；这是策略边界不是密码学边界 |
| FTS 保留明文 token | 中文逐字检索依赖它，加密索引会显著降低召回 |
| 事实卡优先于语义相似 | 改名后旧答案仍可能被向量召回，靠硬优先级纠正 |
| 分角色入库 | 曾经把 user+assistant 拼成一段用 `role=user` 写库，导致助手的话变成"用户证据" |
| 协议门降级为软提醒 | 严格门把简单问题拖成 3–4 次额外工具调用 |

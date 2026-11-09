# dsh-memory-protocol

DeepSeek Harness（DSH）的长期记忆插件：**事实卡（L1）+ 证据时间线（L2）+ 对话流水（L3）+ 区间摘要（L4）**
四层记忆、21 个 MCP 工具、每轮自动注入与自动入库、本地嵌入（零 API 成本）、AES-256-GCM 字段加密。

```
DSH 宿主
 ├─ 本插件（协议层）  pre-step 注入 weave 上下文 / turn-stopping 分角色入库 / 协议门软提醒
 └─ MCP(stdio) → scripts/run-mcp.cjs → python -m memory_skill.mcp_server
      └─ 21 个 memory_* 工具 → SQLite(FTS5) + ChromaDB + bge-large-zh-v1.5(1024d)
```

## 能力一览

| 能力 | 说明 |
|---|---|
| 记忆分层 | L1 事实卡（key-value，可覆盖，值变更进 history）／L2 证据时间线（原话+时间戳，只追加）／L3 对话流水（FTS5 全文 + 向量）／L4 区间摘要（清理前压缩） |
| 作用域 | `world` / `web` / `platform` / `topic:<name>` / `file:<path>` / `session:<name>`，写非默认域自动绑定会话 |
| 检索 | 语义 + BM25 + 时间衰减 + 别名精确命中，RRF 融合；事实卡硬优先；自然语言时间过滤（"上周/昨天"） |
| 写入 | 每轮自动 ingest（user/assistant 分角色）；`distill` 批量提炼候选 → 人工 accept/reject 转正 |
| 加密 | facts / evidence / dialogue / learned 全部 AES-256-GCM（`enc:v1:`），GCM 自带完整性校验 |
| 备份 | 日/周/月/年金字塔，AES 加密，写入即只读，`--restore` 一键回滚 |
| 注入预算 | 1800 字符硬上限 + 优先级填充（超出按优先级裁剪，不会撑爆上下文） |
| 按会话开关 | `memory_toggle` 三态（pause / resume / status），按通道生效，3 秒内生效且不删数据 |
| 图片 | 本轮有图片时注入强提示，保证"看完图会存一张描述"；未存时兜底记占位 |
| 技能层 | `memory_check_skill` / `memory_teach_skill` / `memory_update_skill`，程序记忆独立成腿，检索权重 2.0 |
| 记忆树 | 5 分支（偏好/回忆/人格/任务/技能）× 日期 × 时段，节点在 distill 转正时写入 |
| CLI / 运维 | `python -m memory_skill.cli` + `tools/` 下备份、distill、凭据清洗、加密迁移、换密钥等脚本 |
| 成本 | 本地 ONNX 嵌入，检索零 API 调用；只有 distill（可选）走 LLM |

## 快速开始

### 1. 依赖

```bash
pip install -r requirements.txt          # mcp / chromadb / onnxruntime / numpy / jieba / cryptography
```

**嵌入模型**（约 1.2 GB，不随仓库分发）：下载 `bge-large-zh-v1.5`（或 `bge-large-en-v1.5`），
目录内需有 `model.onnx`，然后：

```bash
export MEMORY_MODEL_PATH=/path/to/bge-large-zh-v1.5     # Windows: $env:MEMORY_MODEL_PATH="D:\models\bge-large-zh-v1.5"
```

未配置模型不会崩，语义检索降级为 BM25（`memory_status` 会显示 `embedder.mode=fallback`）。

### 2. 安装

**一键（Windows）**

```powershell
.\install.ps1 -Profile desktop -ModelPath "D:\models\bge-large-zh-v1.5"
```

**手动**

```bash
# 装进宿主实际在跑的 profile（不确定就别写 web/desktop，先看宿主日志）
dsh plugin --profile <profile> add file:<本仓库绝对路径>
# 重启宿主 —— 插件与 MCP server 只在启动时加载
```

**不使用 DSH、只调后端**

```bash
python -m memory_skill.cli --db ./memory.db status
python -m memory_skill.cli --db ./memory.db ingest "把这段话存进记忆" --role user
python -m memory_skill.cli --db ./memory.db search "怎么备份"
```

### 3. 验证

```bash
python samples/make_sample_db.py --force    # 生成只含示例数据的库
node scripts/doctor.mjs                     # 只读自检（解释器/后端/依赖/模型/库/key）
node test/smoke-mcp.mjs                     # 期望：initialize ✓ / 21 个工具 / embedder=onnx / 4 张事实卡
node test/regress-10.mjs --label baseline   # 10 组查询 + 协议门回归（需语料，示例库命中率不具参考性）
node tools/test-memory-30.mjs               # 30 组端到端检索（自动写入临时库副本）
```

> 两个测试脚本**强制使用 `samples/sample.db`**（或 `MEMORY_SMOKE_DB` 指定的库），
> 不会读写你的真实记忆库。

### 4. 卸载

```bash
dsh plugin --profile <profile> remove dsh-memory-protocol
```

记忆库与密钥不会被删除：`<DSH_HOME>/memory/memory.db`、`master.key`、`memory.db_chroma/`。

## 首次启动会发生什么（新装用户）

**结论：空库，但功能完整。** 没有任何记忆数据，所有工具都能正常调用。

| 首次启动 | 行为 |
|---|---|
| 库文件 | 自动创建 `<DSH_HOME>/memory/memory.db`，自动建表 |
| 密钥 | 自动生成 `<DSH_HOME>/memory/master.key`（0600） |
| 向量库 | 自动创建 `memory.db_chroma/` |
| 工具 | 21 个全部注册可用，`memory_status` → `healthy` |
| 事实卡 | 0 张；`memory_facts` 返回空数组（不是报错） |
| 技能 / 学习队列 / 待审候选 | 全 0 |
| 记忆树 | 0 节点；`weave` 不会注入树概览 |
| `memory_weave` | 返回结构完整，但只有基础提示（无 `[教训]`/`[进行中]`） |
| `memory_ingest` | 立即生效，随后 `memory_search` 能查到 |
| 备份目录 | 不存在，第一次跑 `tools/memory_backup.py` 时创建 |

实测（全新 `DSH_HOME`，无任何数据）：`tools/list` 21 个、`status=healthy`、
`embedder=onnx`、`facts=0`、`weave.text` 26 字符、`ingest` 成功、`db` 143 KB。

用满 1–2 天后典型规模：几十张事实卡 + 上千条对话 + 若干技能；`memory_status`
会显示 `integrity.checked` 与 `learned_store.entry_count`。

## 目录

```
lib/                协议插件（pre-step / turn-stopping / tools-pre-execute）
client/             可选前端组件（记忆开关按钮）
cordis.patch.yml    bundle 补丁：插入 memory-mcp + memory-protocol
python/memory_skill 改造版 Python 后端（43 个模块，随包分发，不依赖 pip 安装的 memory-skill）
scripts/            run-mcp.cjs（启动器）／python-resolve.cjs／doctor.mjs／bootstrap-memory.mjs
test/               smoke-mcp.mjs（冒烟）／regress-10.mjs（回归）
tools/              备份金字塔、distill 日更、凭据清洗、加密迁移、换密钥、30 组检索测试
samples/            make_sample_db.py（只含示例数据）+ 说明
docs/               架构与数据流 + 可选模块用法（OPTIONAL_MODULES.md）
```

## 21 个工具

| 组 | 工具 |
|---|---|
| 检索读写 | `memory_search` `memory_weave` `memory_ingest` `memory_feedback` `memory_status` |
| 事实卡 | `memory_fact_add` `memory_facts` `memory_fact_timeline` |
| 学习闭环 | `memory_classify` `memory_learning_queue` `memory_learning_mark` `memory_distill` `memory_pending` `memory_pending_mark` `memory_conclusions` |
| 技能 | `memory_check_skill` `memory_teach_skill` `memory_update_skill` |
| 维护 | `memory_prune` `memory_episodes` `memory_toggle` |

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSH_HOME` | `~/.dsh` | 宿主数据目录 |
| `MEMORY_SKILL_DB_PATH` | `<DSH_HOME>/memory/memory.db` | 记忆库路径；显式设置时一律尊重 |
| `MEMORY_MODEL_PATH` | 自动探测 | 嵌入模型目录（含 `model.onnx`） |
| `MEMORY_SKILL_PYTHON` | 探测 `python3 → python → py` | 解释器 |
| `MEMORY_PROTOCOL_BACKEND` | 空 | `=installed` 强制用 site-packages 里的 `memory_skill`（排障） |
| `MEMORY_SKIP_BOOTSTRAP` | 空 | `=1` 关闭缺工具自检 |
| `MEMORY_IMPORTANCE_GATE` | `rule` | 设为 `llm` 则每轮 ingest 由 LLM 打重要性分 |
| `IMPORTANCE_API_KEY` | 空 | distill 与 LLM 打分（OpenAI 兼容，默认 DeepSeek） |
| `MEMORY_MOOD_STATE` | 空 | 可选的"情绪状态 JSON"路径，用于给记忆做情绪加权 |
| `OLD_MASTER_KEY_PATH` | 空 | `tools/rekey.py` 用：旧密钥路径 |
| `MEMORY_SMOKE_DB` | `samples/sample.db` | 测试脚本使用的库 |

## 数据与隐私

- 记忆库、ChromaDB 索引、`master.key`、备份、日志**全部在本机**，不含任何向上游的回传。
- 加密边界要诚实：字段级 AES-256-GCM 挡得住 `cat` / `grep` / 备份外泄 / 同机其它账号，
  **挡不住能执行代码的进程**（插件本身要解密才能工作）。密钥与数据同目录是有意为之。
- `dialogue_fts` 的全文索引仍保存明文 token —— 这是"可检索"的代价。
- 仓库本身**不含任何真实记忆**：`samples/sample.db` 由脚本生成，内容全部虚构；
  `.gitignore` 已排除 `memory.db*` / `*.mbk` / `master.key` / `backups/` / 日志。

## 工具脚本（tools/）

```bash
python tools/memory_backup.py                 # 今天的日备份（幂等）
python tools/memory_backup.py --list          # 列出备份
python tools/memory_backup.py --restore <f>   # 恢复（先做一次安全备份）
python tools/daily_distill.py [--force]       # 主动学习日更（带门槛与状态文件）
python tools/purge_secrets.py [--apply]       # 清除误发进库的明文凭据（默认 dry-run）
python tools/encrypt_migrate.py --dry-run     # 历史明文 → 密文（幂等）
python tools/rekey.py --dry-run               # 换密钥/修密钥错配
```

所有脚本的路径来自 `tools/_common.py`：`MEMORY_DIR` / `MEMORY_SKILL_DB_PATH` / `DSH_HOME`，
复制到任何机器都不需要改代码。

## 已知边界

1. **细节记忆窗口**：`dialogue_max_turns=2000` + 日均 200–300 条时，实际细节窗口约 1–2 周，
   更早的对话压成 L4 摘要且不再参与检索。需要长期细节就把上限调大（代价是检索变慢）。
2. **协议门是软提醒**：未分类/未检技能不会拒绝工具调用，只注入一行 `[协议提醒]`。
3. **`file:` 依赖是快照**：改动 `python/` 或 `lib/` 后必须重新 `add` 一次并重启宿主。
4. **新增 MCP 工具**必须同步 `lib/index.js` 的 `MEMORY_TOOL_NAMES` 白名单。
5. **SAW 缓冲**默认空转：没有屏幕采集源时 `ingest_screen` 不会被调用。
6. **四个可选模块**（代理/抓取/合成/直连适配器）不在 21 个工具内，用法见 [docs/OPTIONAL_MODULES.md](docs/OPTIONAL_MODULES.md)。

## License

MIT

# 可选模块（不接 MCP，按需手动使用）

`python/memory_skill/` 里有四个模块**没有被 21 个 MCP 工具覆盖**。它们不是死代码，
而是"另一条入口"，适合脚本化或嵌入其它 agent。默认不分发入口脚本，避免误启动。

| 模块 | 用途 | 依赖 |
|---|---|---|
| `transparent_proxy.py` | OpenAI 兼容 HTTP 代理：请求进来自动注入记忆上下文，响应后自动入库 | `DEEPSEEK_API_KEY`、`DEEPSEEK_API_BASE` |
| `web_crawler.py` | 抓 URL → 清洗正文 → 分块 → 作为对话条目入库 | 无（纯标准库） |
| `knowledge_synth.py` | 多源交叉验证 + LLM 精炼，产出可沉淀的结论 | `IMPORTANCE_API_KEY`（或任意 OpenAI 兼容 key） |
| `room_adapter.py` | 把 `MemorySystem` 包成 `MemoryProtocol`，供 Room Agent 之类宿主直连（不走子进程/MCP） | 无 |

## 1. transparent_proxy —— 给任何 OpenAI 兼容客户端加记忆

```bash
export DEEPSEEK_API_KEY=sk-...
export DEEPSEEK_API_BASE=https://api.deepseek.com/v1   # 可选
export MEMORY_SKILL_DB_PATH=./memory.db
python -m memory_skill.transparent_proxy               # 默认端口见模块内 _DEFAULT_PORT
```

然后把客户端的 `base_url` 指向 `http://127.0.0.1:<port>/v1`。代理会在：
- 请求前：用最后一条 user 消息做 `weave`，把上下文塞进 system/首条消息；
- 响应后：把这一轮按 `user` / `assistant` 分角色 `ingest`。

不配置 key 时脚本直接退出并提示，不会静默启动一个空代理。

## 2. web_crawler —— 把网页变成记忆

```python
import sys; sys.path.insert(0, "python")
from memory_skill.contracts import MemorySkillConfig
from memory_skill._compose import _build_system
from memory_skill.web_crawler import WebCrawler

system = _build_system(MemorySkillConfig(db_path="./memory.db"))
crawler = WebCrawler()

for url in ["https://example.com/article"]:
    chunks = crawler.crawl(url)          # list[CrawledChunk]，失败返回 []
    n = crawler.ingest(system, chunks)   # 作为 DialogueTurn 入库
    print(url, len(chunks), "chunks,", n, "ingested")
```

- 只处理 `text/*` 与 HTML；非文本直接跳过并打日志。
- 抓取失败/解析失败都返回空列表，不抛异常。
- 分块后按普通对话条目录入，因此能立刻被 `memory_search` 检索到。

## 3. knowledge_synth —— 先交叉验证再沉淀

```python
from memory_skill.knowledge_synth import KnowledgeSynth

synth = KnowledgeSynth(api_base="https://api.deepseek.com/v1",
                       api_key="sk-...", model="deepseek-chat")

result = synth.synthesize(topic="如何给 bge 模型换 ONNX 优化",
                          sources=["https://...", "https://..."])
print(result.entries)        # 交叉验证后的事实条目
print(synth.synthesize_markdown("..."))  # 直接产出 markdown 版
```

设计意图：**多源命中才采信**。单源内容只作为候选，不直接写进事实卡——
避免把网页上的错误信息固化成"记忆"。LLM 只做精炼与去重，不做事实裁决。

## 4. room_adapter —— 不用 MCP 的直连方式

```python
from memory_skill.room_adapter import MemorySkillAdapter

adapter = MemorySkillAdapter(agent_name="room-agent", db_path="./memory.db")
ctx = adapter.weave("用户刚才问了什么")     # 直接拿注入文本
adapter.ingest_turn(role="assistant", content="…")
```

适合：宿主已经是一个 Python agent，不想再拉起 stdio 子进程；或在测试里想直接观察
`MemoryProtocol` 的行为。

## 为什么要单独写一页

21 个 MCP 工具覆盖的是"agent 主动调用"的场景。上面四个模块是"由外部流量触发"的场景：
代理靠 HTTP、抓取靠 URL、合成靠多源、适配器靠 Python 直连。它们共享同一套 store，
所以在 A 入口写进去的内容，B 入口能检索到——这也是分层记忆的意义。

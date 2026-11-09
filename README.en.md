# dsh-memory-protocol

Long-term memory plugin for **DeepSeek Harness (DSH)**.

Four memory layers (fact cards L1 + evidence timeline L2 + dialogue stream L3 + episode summaries L4),
21 MCP tools, automatic per-turn context injection and ingestion, local ONNX embeddings
(zero API cost for retrieval) and AES-256-GCM field encryption.

```
DSH host
 ├─ this plugin        pre-step weave injection / turn-stopping role-split ingest / advisory protocol gate
 └─ MCP(stdio) → scripts/run-mcp.cjs → python -m memory_skill.mcp_server
      └─ 21 memory_* tools → SQLite(FTS5) + ChromaDB + bge-large-zh-v1.5 (1024d)
```

## Features

| Area | What you get |
|---|---|
| Layered memory | L1 fact cards (key-value, overwrite with history) / L2 evidence timeline (verbatim quotes) / L3 dialogue stream (FTS5 + vectors) / L4 episode summaries |
| Scopes | `world` `web` `platform` `topic:<name>` `file:<path>` `session:<name>`, auto-bound on write |
| Retrieval | semantic + BM25 + recency decay + alias exact-match, fused with RRF; fact cards take precedence; natural-language time filters |
| Ingestion | automatic per turn (user/assistant kept apart); `distill` proposes candidates, human accepts/rejects |
| Encryption | facts / evidence / dialogue / learned docs are AES-256-GCM (`enc:v1:`), GCM integrity built in |
| Backup | daily/weekly/monthly/yearly pyramid, encrypted, write-once, `--restore` |
| Injection budget | 1800 chars hard cap with priority filling |
| Channel switch | `memory_toggle` pause/resume/status, effective in 3 s, never deletes data |
| Images | strong hint when a turn contains images; placeholder insert if no description was stored |
| Skills | `memory_check_skill` / `memory_teach_skill` / `memory_update_skill`, separate retrieval leg (weight 2.0) |
| Memory tree | 5 branches × date × period, populated when candidates are promoted |
| CLI & ops | `python -m memory_skill.cli` plus backup / distill / secret-purge / migration tools under `tools/` |

## Quick start

```bash
pip install -r requirements.txt

# optional but recommended: local embedding model (~1.2 GB, not shipped)
export MEMORY_MODEL_PATH=/path/to/bge-large-zh-v1.5

# install into the profile your host actually runs, then restart the host
dsh plugin --profile <profile> add file:<absolute-path-of-this-repo>

# verification
python samples/make_sample_db.py --force
node scripts/doctor.mjs
node test/smoke-mcp.mjs        # expect: 21 tools / embedder=onnx / 4 fact cards
node tools/test-memory-30.mjs  # 30 end-to-end retrieval cases on a temp copy
```

Standalone backend (no DSH):

```bash
python -m memory_skill.cli --db ./memory.db status
python -m memory_skill.cli --db ./memory.db search "how do I back this up"
```

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `DSH_HOME` | `~/.dsh` | host data directory |
| `MEMORY_SKILL_DB_PATH` | `<DSH_HOME>/memory/memory.db` | memory database |
| `MEMORY_MODEL_PATH` | auto-probe | embedding model dir containing `model.onnx` |
| `MEMORY_SKILL_PYTHON` | probe `python3 → python → py` | interpreter |
| `MEMORY_IMPORTANCE_GATE` | `rule` | set `llm` to score every ingest with an LLM |
| `IMPORTANCE_API_KEY` | empty | required for `distill` / LLM scoring |
| `MEMORY_SMOKE_DB` | `samples/sample.db` | database used by test scripts |

## Privacy

Everything stays local: database, Chroma index, `master.key`, backups and logs.
Encryption is field-level: it protects against `cat` / `grep` / backup exfiltration, not against a
process that can execute code (the plugin must decrypt to work). The FTS index keeps plaintext
tokens by design — that is the price of full-text search.

This repository ships **no real memories**: `samples/sample.db` is generated from fictional data and
`memory.db*` / `*.mbk` / `master.key` / `backups/` / logs are git-ignored.

## Known limits

1. Detailed recall window is bounded by `dialogue_max_turns` (default 2000); older turns collapse into L4 summaries.
2. The protocol gate is advisory: it never blocks tool calls.
3. `file:` dependencies are snapshots — re-`add` and restart after editing `python/` or `lib/`.
4. New MCP tools must be added to `MEMORY_TOOL_NAMES` in `lib/index.js`.
5. The SAW ring buffer stays idle when no screen-capture source exists.

## License

MIT

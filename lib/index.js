/**
 * Memory protocol enforcement plugin.
 *
 * Mirrors the opencode-side `solo-memory` hard protocol on DeepSeek Harness:
 *
 * 1. Force `memory_weave` once per agent turn before any non-memory tool may run.
 *    `tools/pre-execute` denies every tool call while the turn's weave is pending;
 *    the memory tools themselves (`mcp__opencode_memory__*`) are whitelisted so the
 *    weave can actually happen.
 * 2. Auto-inject the weave context into the first step of each turn via
 *    `agent/pre-step`, so the model sees the relevant memory without asking.
 * 3. Auto-ingest the completed turn at `agent/turn-stopping` from the buffered
 *    `session/event` stream, so every dialogue block is persisted without the
 *    model having to call the tool itself.
 * 4. Fail-open when the memory MCP tools are missing (a marketplace install
 *    that skipped the Python side): the gate passes instead of paralyzing the
 *    agent, a one-time loud guidance is surfaced, and the bootstrap script
 *    (scripts/bootstrap-memory.mjs) is spawned once to install the Python deps
 *    + embedding model. Strict mode resumes automatically the moment the
 *    tools register.
 *
 * The memory MCP server is reached through the registered MCP-bridged tools
 * (`ctx.tools.execute({ name: 'mcp__opencode_memory__memory_weave', ... })`), so this
 * plugin has no direct dependency on the Python server; it only requires the
 * `mcp__opencode_memory__` tools to be present (mounted via the mcp-client overlay).
 *
 * @module @deepseek-ai/dsh-memory-protocol
 */
import Schema from '@deepseek-ai/schemastery';
import { ToolCallId, createUserMessage } from '@deepseek-ai/dsh-llm';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
/** MCP-bridged tool name prefix for the opencode memory server. */
const MCP_PREFIX = 'mcp__opencode_memory__';
/** Raw tool names the protocol must allow through (the memory tools themselves). */
const MEMORY_TOOL_NAMES = new Set([
    'memory_weave',
    'memory_ingest',
    'memory_status',
    'memory_search',
    'memory_feedback',
    'memory_classify',
    'memory_teach_skill',
    'memory_update_skill',
    'memory_check_skill',
    'memory_learning_queue',
    'memory_learning_mark',
    'memory_distill',
    'memory_pending',
    'memory_pending_mark',
    'memory_conclusions',
    // 0.4.0：事实卡（L1）与证据时间线（L2），以及 L3/L4 清理
    'memory_fact_add',
    'memory_facts',
    'memory_fact_timeline',
    'memory_prune',
    'memory_episodes',
    // 六项增强：记忆开关（按通道/全局暂停）
    'memory_toggle',
]);
/** Message source marking protocol-injected context as producer-owned (log-rebuildable).
 * Session format v4 rejects the retired V3 wrapper {kind:'plugin'} and requires a
 * producer-owned kind; `plugin:memory-protocol` is the exact kind the v3→v4 migration
 * assigns to this plugin, so old and new sessions agree. */
const PLUGIN_SOURCE = { kind: 'plugin:memory-protocol' };
/** One-time guidance surfaced when the memory MCP tools are missing. */
const MISSING_TOOLS_GUIDANCE = 'memory-protocol: memory MCP tools NOT registered (内置 memory_skill 后端未启动)。' +
    '排错: node node_modules/dsh-memory-protocol/scripts/doctor.mjs；常见原因：' +
    '1) Python 解释器不对 —— 设 MEMORY_SKILL_PYTHON 指向真实 python.exe；' +
    '2) 缺运行时依赖（mcp / chromadb / onnxruntime / numpy / jieba）；' +
    '3) 嵌入模型目录不存在（设 MEMORY_MODEL_PATH）。' +
    'Agent is running WITHOUT memory until the tools appear.';
/** One-time activation card injected once the memory tools ARE registered. */
const ACTIVATION_GUIDANCE = '[memory-protocol] 记忆系统已激活（memory_weave 等 21 个工具可用，后端为插件内置 0.4.1 版）。\n' +
    '• 状态/详情: 调 memory_status 查看（embedder 模式、库位置、记忆条目数）。\n' +
    '• 上下文已自动注入: 每轮 pre-step 插件会自己调 memory_weave 并注入结果，turn-stopping 自动 ingest 本轮用户与助手消息——不需要你重复调这两个工具。\n' +
    '• 事实卡（L1）: 用户明确声明身份/偏好时用 memory_fact_add(category, key, value) 落卡；盘点用 memory_facts(category)。\n' +
    '• 证据时间线（L2）: memory_fact_timeline(key_or_id) 返回当前值、历史值、原话证据；对话流水清理不影响它。\n' +
    '• 主动检索: 注入的上下文不够时再用 memory_search(query)。\n' +
    '• 分类是可选的: memory_classify 供主动学习闭环使用，不分类不会再阻断任何工具调用（协议门 0.4.1 起降级为软提醒）。\n' +
    '• distill（批量提炼候选卡）需要配置 IMPORTANCE_API_KEY（写入 ~/.config/memory-skill/.env 或项目 .env）；未配置时 distill 不可用，但可直接用 memory_ingest / memory_teach_skill 沉淀。\n' +
    '• 排错: memory_status 里 embedder.mode=fallback 表示模型缺失（语义检索降级为 BM25）。';
/** One-time ask-for-key instruction injected alongside the activation card when
 * the importance LLM key is missing. Tells the agent to proactively request the
 * key from the user so distill/active-learning work from the start, instead of
 * surfacing only when the user happens to call distill. */
const ASK_KEY_GUIDANCE = '⚠️ 请确认主动学习（memory_distill）的 API key：先调 memory_status 查看 importance_llm.configured。\n' +
    '  • 若 configured=false: 主动向用户索要 DeepSeek API key（一次即可，用户拒绝/没有就跳过，不要反复追问）：\n' +
    '      1. 说明用途: 用于 memory_distill 批量提炼记忆候选卡时给内容打重要性分（可选功能，记忆核心不受影响）。\n' +
    '      2. 索要内容: platform.deepseek.com 申请的 sk- 开头 API key。\n' +
    '      3. 用户提供则帮他写入 ~/.config/memory-skill/.env（IMPORTANCE_API_KEY=sk-xxx），重启 harness 生效。\n' +
    '  • 若 configured=true: 已配置，跳过此步骤即可。';
/** Full one-time activation payload: base guidance + the ask-for-key instruction
 * (only surfaces on the first activation, gated by introInjected). */
function fullActivationGuidance() {
    return ACTIVATION_GUIDANCE + '\n\n' + ASK_KEY_GUIDANCE;
}
// Fail-open bootstrap state — module-level, so each surface fires once per process.
let warnedMissing = false; // logger.warn guidance emitted once while tools are missing
let guidanceInjected = false; // guidance user-message injected once
let bootstrapSpawned = false; // bootstrap script spawned at most once
let resumedLogged = false; // transition log once when tools come back
let introInjected = false; // activation card injected once when tools are available
/**
 * Per-agent protocol state: whether the CURRENT turn has already run its weave.
 * Reset to `false` at `agent/turn-stopping` so a fresh turn must weave again.
 */
const weavedThisTurn = new WeakMap();
/** Per-session buffered text of user/assistant messages, flushed at turn-stopping. */
const turnBuffers = new WeakMap();
/** 本轮出现的图片 attachment id（pre-step 记，turn-stopping 兜底存描述）。 */
const imagesThisTurn = new WeakMap();
/** Returns true when the tool name is one of the memory protocol's own tools. */
function isMemoryTool(name) {
    if (!name.startsWith(MCP_PREFIX))
        return false;
    return MEMORY_TOOL_NAMES.has(name.slice(MCP_PREFIX.length));
}
/** The MCP-bridged public name for a raw memory tool name. */
function memoryTool(raw) {
    return `${MCP_PREFIX}${raw}`;
}
/**
 * Dynamic check: memory tools may register after apply (MCP connect, restart,
 * reconnect), so this must be evaluated per hook invocation, never cached.
 */
function memoryToolsAvailable(ctx) {
    return ctx.tools.get(memoryTool('memory_weave')) !== undefined;
}
/** Emit the loud missing-tools guidance to the log, at most once per process. */
function warnMissingOnce(ctx) {
    if (warnedMissing)
        return;
    warnedMissing = true;
    ctx.logger.warn(MISSING_TOOLS_GUIDANCE);
}
/** Log the missing → present transition once (strict gate resumes). */
function noteResumedOnce(ctx) {
    if (!warnedMissing || resumedLogged)
        return;
    resumedLogged = true;
    ctx.logger.warn('memory-protocol: memory MCP tools registered — strict gate resumed');
}
/**
 * Spawn the bootstrap script (pip install + model download) at most once per
 * process, non-blocking. A spawn failure must never throw out of a hook.
 */
function maybeSpawnBootstrap(ctx, config) {
    if (bootstrapSpawned)
        return;
    bootstrapSpawned = true;
    try {
        const script = fileURLToPath(new URL('../scripts/bootstrap-memory.mjs', import.meta.url));
        spawn(process.execPath, [script], { detached: false, stdio: 'inherit', env: process.env });
        ctx.logger.warn('memory-protocol: bootstrap started — pip install + model download; restart dsh after it completes to mount the MCP server');
    }
    catch (error) {
        ctx.logger.warn(`memory-protocol: bootstrap spawn failed: ${String(error)}`);
    }
}
/** 记忆分区判定：QQ/平台会话读写 platform 区，Web 会话用 web 区。
 *
 * 2026-10-28: 此前 weave/ingest 都没有传 scope，所有渠道的对话都落到默认
 * web 区 —— QQ 会话读不到自己刚聊过的内容，proactive_chat.py 也读不到真实
 * 对话（拿 12 小时前的旧记录当"最近"）。session id 前缀 `onebot-` 与
 * agentPreset `platform-chat` 是平台渠道的稳定标识。
 */
function memoryScopeOf(agent) {
    const session = agent && agent.session;
    const id = String((session && session.id) || '');
    if (id.startsWith('onebot-'))
        return 'platform';
    const preset = String((session && (session.agentPreset || session.header?.agentPreset)) || '');
    return preset === 'platform-chat' ? 'platform' : 'web';
}
/** 按通道/会话暂停记忆：paused.json 形如 {"global":false,"scopes":["platform"]}。
 *  文件位置优先跟 MEMORY_SKILL_DB_PATH 同目录，其次 DSH_HOME/memory。
 *  关闭时：pre-step 不注入、turn-stopping 不写入（数据一条不删）。 */
function pauseFilePath() {
    const db = process.env.MEMORY_SKILL_DB_PATH;
    if (db)
        return path.join(path.dirname(db), 'paused.json');
    const home = process.env.DSH_HOME;
    if (home)
        return path.join(home, 'memory', 'paused.json');
    return path.join(os.homedir(), '.dsh', 'memory', 'paused.json');
}
let _pauseCache = { at: 0, data: null };
function memoryPaused(agent) {
    const now = Date.now();
    if (now - _pauseCache.at > 3000) {
        _pauseCache.at = now;
        try {
            _pauseCache.data = JSON.parse(fs.readFileSync(pauseFilePath(), 'utf8'));
        }
        catch {
            _pauseCache.data = null;
        }
    }
    const cfg = _pauseCache.data;
    if (!cfg)
        return false;
    if (cfg.global === true)
        return true;
    const scopes = Array.isArray(cfg.scopes) ? cfg.scopes.map(String) : [];
    return scopes.includes(memoryScopeOf(agent));
}
/** Invoke a memory MCP tool through the registered ToolRuntime. */
async function callMemoryTool(ctx, agent, signal, name, args) {
    const input = {
        callId: ToolCallId(`memory-protocol-${name}-${Math.random().toString(16).slice(2, 10)}`),
        name: `${MCP_PREFIX}${name}`,
        arguments: args,
        signal,
        ...(agent === undefined ? {} : { agent }),
    };
    const result = await ctx.tools.execute(input);
    if (result.isError) {
        throw new Error(`memory tool ${name} failed: ${String(result.error)}`);
    }
    return result.value;
}
/** Extract a handler-level error out of a memory tool result, if any. */
function memoryToolError(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value))
        return '';
    const error = value.error;
    if (typeof error === 'string')
        return error;
    // DSH mcp-client wraps results as { content: [{ type, text: <json> }] }.
    const content = value.content;
    if (Array.isArray(content)) {
        for (const block of content) {
            const raw = block && typeof block === 'object' ? block.text : undefined;
            if (typeof raw !== 'string' || !raw.trim().startsWith('{'))
                continue;
            try {
                const parsed = JSON.parse(raw);
                if (parsed && typeof parsed === 'object' && typeof parsed.error === 'string')
                    return parsed.error;
            }
            catch {
                // not JSON — not an error payload
            }
        }
    }
    return '';
}
/** Decode a weave payload: the Python tool returns a JSON string whose
 *  model-visible field is ``text`` (with ``prompt_block`` as a fallback). */
function coerceWeavePayload(raw) {
    if (typeof raw !== 'string')
        return '';
    const trimmed = raw.trim();
    if (!trimmed)
        return '';
    if (!trimmed.startsWith('{') && !trimmed.startsWith('['))
        return trimmed;
    try {
        const parsed = JSON.parse(trimmed);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            if (typeof parsed.error === 'string')
                return '';
            const text = parsed.text;
            if (typeof text === 'string' && text.trim())
                return text;
            const block = parsed.prompt_block;
            if (typeof block === 'string' && block.trim())
                return block;
        }
    }
    catch {
        // not JSON after all — fall through to the raw text
    }
    return trimmed;
}
/** Extract the model-visible text from the weave result. Handles the DSH
 *  mcp-client wrapper ``{ content: [{ type: 'text', text: <json> }] }``
 *  (0.2.x) as well as the older flat ``{ text }`` / ``{ context }`` shapes. */
function weaveContextText(value) {
    if (typeof value === 'string')
        return coerceWeavePayload(value);
    if (Array.isArray(value)) {
        const joined = value
            .map((block) => (block && typeof block === 'object' ? block.text : ''))
            .filter((text) => typeof text === 'string')
            .join('\n');
        return coerceWeavePayload(joined);
    }
    if (value && typeof value === 'object') {
        const content = value.content;
        if (Array.isArray(content)) {
            const fromContent = weaveContextText(content);
            if (fromContent)
                return fromContent;
        }
        if (value.structuredContent !== undefined) {
            const fromStructured = weaveContextText(value.structuredContent);
            if (fromStructured)
                return fromStructured;
        }
        const text = value.text;
        if (typeof text === 'string')
            return text;
        const context = value.context;
        if (typeof context === 'string')
            return context;
        if (Array.isArray(context)) {
            return context
                .map((block) => (block && typeof block === 'object' ? block.text : ''))
                .filter((text) => typeof text === 'string')
                .join('\n');
        }
    }
    return '';
}
/** Append one session event's text to its turn buffer (role-tagged). */
function bufferEventText(session, event) {
    if (event.type !== 'user/message' && event.type !== 'assistant/message')
        return;
    const role = event.type === 'user/message' ? 'user' : 'assistant';
    const data = event.data;
    const content = data.message?.content ?? [];
    const text = content
        .filter((block) => block.type === 'text')
        .map((block) => block.text ?? '')
        .join('\n');
    if (!text.trim())
        return;
    const buffer = turnBuffers.get(session) ?? [];
    // The pre-step seed and the session/event stream can both deliver the
    // same user text; keep one copy so a turn is not stored twice.
    const last = buffer[buffer.length - 1];
    if (last && last.role === role && last.text === text)
        return;
    buffer.push({ role, text });
    turnBuffers.set(session, buffer);
}
export const name = 'memory-protocol';
export const inject = ['tools'];
export const Config = Schema.object({
    enforceWeave: Schema.boolean().default(true),
    injectWeave: Schema.boolean().default(true),
    autoIngest: Schema.boolean().default(true),
    allowlist: Schema.array(Schema.string()).default([]),
    bootFailOpen: Schema.boolean().default(true),
    autoBootstrap: Schema.boolean().default(true),
    autoBackup: Schema.boolean().default(false),
});
export function apply(ctx, config) {
    const allow = new Set(config.allowlist);
    const bufferingAgents = new WeakSet();
    // ── 客户端胶囊按钮：记忆开关 HTTP 路由（同源，读写 paused.json）────────
    // 客户端 bundle 通过 GET/POST /dsh-memory/toggle 读取并切换开关；
    // 写的是同一个 paused.json，pre-step / turn-stopping 的 memoryPaused()
    // 会在 3 秒缓存窗口内读到新值。
    ctx.inject(['webServer'], (host) => {
        const dshHome = process.env.DSH_HOME || path.join(os.homedir(), '.dsh');
        const pauseFile = path.join(dshHome, 'memory', 'paused.json');
        host.effect(() => host.webServer.register({
            kind: 'exact',
            path: '/dsh-memory/toggle',
            handler: async (request, response) => {
                let data = { global: false, scopes: [] };
                try {
                    data = JSON.parse(fs.readFileSync(pauseFile, 'utf8'));
                }
                catch {
                    // 文件不存在 = 未暂停，保持默认
                }
                if (request.method === 'POST') {
                    let body = '';
                    await new Promise((resolve) => {
                        request.on('data', (chunk) => {
                            body += chunk;
                        });
                        request.on('end', resolve);
                    });
                    let payload = {};
                    try {
                        payload = JSON.parse(body || '{}');
                    }
                    catch {
                        // 坏 body 当作 toggle
                    }
                    const scope = String(payload.scope || 'all');
                    const scopes = Array.isArray(data.scopes) ? data.scopes.map(String) : [];
                    const curPaused = Boolean(data.global) || (scope !== 'all' && scopes.includes(scope));
                    const action = String(payload.action || 'toggle');
                    const on = action === 'toggle' ? !curPaused : action === 'pause';
                    if (scope === 'all')
                        data.global = on;
                    else {
                        const set = new Set(scopes);
                        if (on)
                            set.add(scope);
                        else
                            set.delete(scope);
                        data.scopes = [...set].sort();
                    }
                    try {
                        fs.mkdirSync(path.dirname(pauseFile), { recursive: true });
                        fs.writeFileSync(pauseFile, JSON.stringify(data));
                    }
                    catch {
                        // 写失败维持原状，客户端下一次轮询会纠正
                    }
                }
                const scopes = Array.isArray(data.scopes) ? data.scopes : [];
                const paused = Boolean(data.global) || scopes.length > 0;
                response.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
                response.end(JSON.stringify({ paused, global: Boolean(data.global), scopes }));
            },
        }), 'dsh-memory: toggle route');
    });
    // session/event is scope-filtered to the agent's own context, so the buffer
    // listener must live on agent.ctx, not the root ctx. Register once per agent
    // on first sight so user/assistant text is captured for turn ingest.
    function ensureBuffering(agent) {
        if (bufferingAgents.has(agent))
            return;
        bufferingAgents.add(agent);
        agent.ctx.on('session/event', (session, event) => {
            bufferEventText(session, event);
        });
    }
    // tools/pre-execute: the protocol gate. Deny non-memory tools while the
    // current turn's weave is still pending.
    ctx.on('tools/pre-execute', async (exec, next) => {
        if (memoryToolsAvailable(ctx))
            noteResumedOnce(ctx);
        if (exec.name === memoryTool('memory_weave')) {
            // An explicit model-driven weave also satisfies the gate for this turn.
            if (exec.agent)
                weavedThisTurn.set(exec.agent, true);
            return next();
        }
        if (isMemoryTool(exec.name) || allow.has(exec.name))
            return next();
        const agent = exec.agent;
        if (agent !== undefined && weavedThisTurn.get(agent))
            return next();
        // Fail-open: without the memory tools the gate can never be satisfied —
        // let every non-memory tool through and surface the guidance once.
        if (config.bootFailOpen && !memoryToolsAvailable(ctx)) {
            warnMissingOnce(ctx);
            return next();
        }
        if (!config.enforceWeave)
            return next();
        return {
            kind: 'deny',
            reason: 'Memory protocol: call mcp__opencode_memory__memory_weave first, ' +
                'then retry this tool. Every turn must consult memory before acting.',
        };
    });
    // agent/pre-step: weave once per turn and inject the context into the step.
    ctx.on('agent/pre-step', async ({ agent, messages, signal }, next) => {
        if (memoryPaused(agent)) {
            // 记忆开关关闭：不注入、不要求 weave。
            return next();
        }
        ensureBuffering(agent);
        // The first pre-step sees the turn's user messages; the session/event
        // listener (registered just above) only captures events after this point,
        // so seed the buffer with the user text to guarantee it is ingested.
        const session = agent.session;
        const buffer = turnBuffers.get(session) ?? [];
        if (buffer.length === 0) {
            const userText = messages
                .map((message) => message.content
                .filter((block) => block.type === 'text')
                .map((block) => block.text ?? '')
                .join(' '))
                .filter(Boolean)
                .join('\n');
            if (userText) {
                buffer.push({ role: 'user', text: userText });
                turnBuffers.set(session, buffer);
            }
        }
        const downstream = await next();
        if (downstream.kind !== 'enter')
            return downstream;
        if (weavedThisTurn.get(agent))
            return downstream;
        if (!memoryToolsAvailable(ctx)) {
            if (config.bootFailOpen) {
                // Fail-open: skip the auto-weave branch entirely, spawn the bootstrap
                // once, and inject a one-time guidance message. The gate is open, so
                // weavedThisTurn is deliberately left unset.
                if (config.autoBootstrap)
                    maybeSpawnBootstrap(ctx, config);
                if (!guidanceInjected) {
                    guidanceInjected = true;
                    const injected = createUserMessage({
                        content: [{ type: 'text', text: MISSING_TOOLS_GUIDANCE }],
                        source: { ...PLUGIN_SOURCE, form: 'instructions' },
                    });
                    return { kind: 'enter', messages: [injected, ...downstream.messages] };
                }
                return downstream;
            }
            if (config.injectWeave) {
                // Fail-closed: the gate stays closed until the tools appear.
                ctx.logger.warn('memory-protocol: memory_weave tool not registered yet; protocol gate stays closed');
            }
            return downstream;
        }
        if (!config.injectWeave) {
            // Without auto-weave the gate stays closed until the model weaves on its own.
            if (!introInjected) {
                introInjected = true;
                const injected = createUserMessage({
                    content: [{ type: 'text', text: fullActivationGuidance() }],
                    source: { ...PLUGIN_SOURCE, form: 'instructions' },
                });
                return { kind: 'enter', messages: [injected, ...downstream.messages] };
            }
            return downstream;
        }
        const userText = messages
            .map((message) => message.content
            .filter((block) => block.type === 'text')
            .map((block) => block.text ?? '')
            .join(' '))
            .join('\n')
            .slice(0, 2000);
        // 图片记忆：本轮有图片时注入强提示，保证 agent 看完图会存"这张图讲了什么"。
        const imageIds = messages
            .flatMap((message) => message.content || [])
            .filter((block) => block.type === 'image')
            .map((block) => (block.source && (block.source.attachmentId || block.source.id)) ||
            block.attachmentId || '')
            .filter(Boolean);
        imagesThisTurn.set(agent, imageIds);
        const imageHint = imageIds.length
            ? `\n\n[图片记忆] 本轮有 ${imageIds.length} 张图片（attachment id: ${imageIds.slice(0, 3).join(', ')}）。看完后用 memory_ingest 存一条 \`[图片] <id>：这张图讲了什么\`（role=assistant），之后能记住图的内容。`
            : '';
        let context = '';
        let value;
        let weaveError = '';
        try {
            value = await callMemoryTool(ctx, agent, signal, 'memory_weave', { user_message: userText, scope: memoryScopeOf(agent) });
        }
        catch (error) {
            weaveError = String(error);
        }
        // The Python protocol gate used to REJECT the auto-weave (unclassified
        // previous turn / mission skill-check), which left the gate closed and
        // pushed 3–4 extra tool calls onto the model. 0.4.1 made the gate
        // advisory; this branch is the compatibility path for an older backend
        // — heal it here instead of handing the mess to the agent.
        if (/ClassificationRequired|SkillCheckRequired|GapRequired/.test(weaveError) ||
            /ClassificationRequired|SkillCheckRequired|GapRequired/.test(memoryToolError(value))) {
            ctx.logger.warn('memory-protocol: auto-weave hit the strict gate — healing with memory_classify(chat) and retrying once');
            try {
                await callMemoryTool(ctx, agent, signal, 'memory_classify', { category: 'chat' });
                value = await callMemoryTool(ctx, agent, signal, 'memory_weave', { user_message: userText, scope: memoryScopeOf(agent) });
                weaveError = '';
            }
            catch (retryError) {
                weaveError = String(retryError);
            }
        }
        const handlerError = weaveError || memoryToolError(value);
        if (handlerError) {
            // Fail-closed: a failed weave leaves the gate closed; the model must retry.
            ctx.logger.warn(`memory-protocol: memory_weave failed: ${handlerError}`);
            return downstream;
        }
        context = weaveContextText(value);
        if (!context) {
            // A successful but empty weave still counts as having consulted memory.
            // Log it: an empty context here means extraction failed (the MCP bridge
            // wraps results as { content: [{ type: 'text', text: <json> }] }), which
            // used to make injection fail silently for every turn.
            ctx.logger.warn(`memory-protocol: memory_weave returned no injectable text (keys=${value && typeof value === 'object' ? Object.keys(value).join(',') : typeof value})`);
            weavedThisTurn.set(agent, true);
            return downstream;
        }
        weavedThisTurn.set(agent, true);
        const intro = introInjected ? '' : ((introInjected = true), fullActivationGuidance() + '\n\n');
        const injected = createUserMessage({
            content: [{ type: 'text', text: intro + context.slice(0, 8000) + imageHint }],
            source: { ...PLUGIN_SOURCE, form: 'instructions' },
        });
        return { kind: 'enter', messages: [injected, ...downstream.messages] };
    });
    // agent/turn-stopping: ingest the finished turn and reset the weave gate.
    ctx.on('agent/turn-stopping', async ({ agent }) => {
        weavedThisTurn.set(agent, false);
        if (memoryPaused(agent)) {
            turnBuffers.delete(agent.session);
            imagesThisTurn.delete(agent);
            return;
        }
        if (!config.autoIngest)
            return;
        const session = agent.session;
        if (!memoryToolsAvailable(ctx)) {
            // No tools to call — drop this turn's buffer instead of ingesting.
            turnBuffers.delete(session);
            imagesThisTurn.delete(agent);
            return;
        }
        const buffer = turnBuffers.get(session) ?? [];
        turnBuffers.delete(session);
        if (buffer.length === 0)
            return;
        // Ingest each side of the turn with its own role. The old code joined
        // user+assistant into one blob and posted it as role='user', which
        // stored every assistant reply as if the user had said it.
        for (const entry of buffer) {
            const content = String(entry.text ?? '').slice(0, 4000);
            if (!content)
                continue;
            try {
                await callMemoryTool(ctx, agent, new AbortController().signal, 'memory_ingest', { content, role: entry.role, scope: memoryScopeOf(agent) });
            }
            catch {
                // A failed ingest must not block the turn from closing.
            }
        }
        // 图片兜底：本轮有图但 agent 没存描述时，至少记一条占位，
        // 避免原图被清理后完全失忆。
        const imgs = imagesThisTurn.get(agent) || [];
        imagesThisTurn.delete(agent);
        if (imgs.length) {
            const hasImageNote = buffer.some((entry) => String(entry.text ?? '').includes('[图片]'));
            if (!hasImageNote) {
                try {
                    await callMemoryTool(ctx, agent, new AbortController().signal, 'memory_ingest', {
                        content: `[图片] ${imgs.join(', ')}：用户本轮发了 ${imgs.length} 张图片（未记录描述）`,
                        role: 'assistant',
                        scope: memoryScopeOf(agent),
                    });
                }
                catch {
                    // 兜底失败不阻塞收尾。
                }
            }
        }
        if (config.autoBackup) {
            signalBackup(agent.session);
        }
    });
}

// ── Cloud backup signal (ccmp-backup daemon, debounced) ──────────────────
// Best-effort append after a finished turn; the daemon debounces so a
// backup fires only when the user goes idle. Fail-open: never throw.
function signalBackup(_session) {
    try {
        const envVal = process.env.SOLO_MEMORY_BACKUP_SIGNAL;
        if (envVal === '0') return;
        const signalPath = envVal || path.join(os.homedir(), '.ccmp-backup', 'signal');
        fs.appendFileSync(signalPath, `${new Date().toISOString()}\n`);
    }
    catch { /* fail-open: backup signal is best-effort */ }
}

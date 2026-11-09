#!/usr/bin/env node
/**
 * 10 组记忆查询 + 协议门回归测试（直接拉起内置后端的 MCP server，不经过 dsh）。
 *
 *   node test/regress-10.mjs --label baseline --root <pluginRoot> [--db <path>]
 *
 * 输出：
 *   1. 10 组查询的 top-1 命中情况（准确率）
 *   2. 协议门场景 A–D（跨轮 classify 门 / mission skill-check 门）
 *   3. 同一句话的入库次数（weave 写副作用）与 assistant 角色是否保真
 *
 * 每次都把库复制到临时目录，不污染真实记忆库。
 */
import { spawn, spawnSync } from 'node:child_process'
import { createInterface } from 'node:readline'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const argv = process.argv.slice(2)
function arg(name, fallback) {
  const i = argv.indexOf(`--${name}`)
  return i >= 0 ? argv[i + 1] : fallback
}

const here = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(arg('root', path.resolve(here, '..')))
const label = arg('label', 'run')
const sourceDb = arg('db', process.env.MEMORY_SMOKE_DB || path.join(root, 'samples', 'sample.db'))
const launcher = path.join(root, 'scripts', 'run-mcp.cjs')

if (!fs.existsSync(launcher)) fail(`找不到 ${launcher}`)

// ── 临时库副本 ────────────────────────────────────────────────────────────────
const workDir = fs.mkdtempSync(path.join(os.tmpdir(), `mem-regress-${label}-`))
const dbPath = path.join(workDir, 'memory.db')
fs.copyFileSync(sourceDb, dbPath)
const chromaSrc = sourceDb + '_chroma'
if (fs.existsSync(chromaSrc)) {
  fs.cpSync(chromaSrc, dbPath + '_chroma', { recursive: true })
}

// ── MCP stdio 客户端 ─────────────────────────────────────────────────────────
const child = spawn(process.execPath, [launcher], {
  cwd: root,
  env: { ...process.env, MEMORY_SKILL_DB_PATH: dbPath, PYTHONDONTWRITEBYTECODE: '1' },
  stdio: ['pipe', 'pipe', 'pipe'],
})
const rl = createInterface({ input: child.stdout })
const pending = new Map()
let stderrTail = ''
child.stderr.on('data', (c) => { stderrTail = (stderrTail + c.toString()).slice(-6000) })

rl.on('line', (line) => {
  let message
  try { message = JSON.parse(line) } catch { return }
  const id = message.id
  if (id !== undefined && pending.has(id)) {
    const { resolve, reject, timer, method } = pending.get(id)
    clearTimeout(timer)
    pending.delete(id)
    if (message.error) reject(new Error(`${method}: ${message.error.message ?? JSON.stringify(message.error)}`))
    else resolve(message)
  }
})

let nextId = 1
function request(method, params, timeoutMs = 180_000) {
  const id = nextId++
  child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`${method} 超时`)) }, timeoutMs)
    pending.set(id, { resolve, reject, timer, method })
  })
}
function notify(method, params) {
  child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n')
}

/** tools/call，返回解析后的 JSON（handler 返回的 dict）。 */
async function call(name, args = {}) {
  const res = await request('tools/call', { name, arguments: args })
  const text = res.result?.content?.[0]?.text ?? ''
  try { return JSON.parse(text) } catch { return { raw: text } }
}

function sqlite(script) {
  const out = spawnSync('python', ['-c', script], { encoding: 'utf8' })
  if (out.status !== 0) throw new Error(`sqlite helper failed: ${out.stderr}`)
  return JSON.parse(out.stdout.trim())
}

function fail(message) {
  console.error(`\n✗ ${message}`)
  if (stderrTail) console.error(`\n--- MCP stderr（末尾）---\n${stderrTail}`)
  try { child.kill() } catch {}
  process.exit(1)
}

// ── 10 组查询（期望值来自真实库中的事实卡/对话） ─────────────────────────────
const QUERIES = [
  { q: '我家猫叫什么', expectId: 'fact:pers:cat_name' },
  { q: '我叫什么名字', expectId: 'fact:pers:name' },
  { q: '记忆库文件放在哪个路径', expectId: 'fact:pref:memory_db_path' },
  { q: '我要求用什么语言推理', expectId: 'fact:pref:reasoning_language' },
  { q: 'dsh-memory-protocol 这个插件是怎么安装生效的', expectId: 'fact:conclusion:memory_protocol_plugin' },
  { q: 'SAW 是什么，怎么写入的', expectId: 'fact:conclusion:saw_buffer' },
  { q: '记忆树是怎么组织和分层的', expectId: 'fact:conclusion:memory_tree' },
  { q: 'approval policy 从 ask 改成了 never', expectContains: 'approval' },
  { q: '量子纠缠的贝尔不等式怎么推导', expectNoStrong: 0.45 },
  { q: '红烧肉要炖多久才软烂', expectNoStrong: 0.45 },
]

function matchQuery(entry, spec) {
  if (spec.expectNoStrong !== undefined) {
    if (!entry) return true // 无关查询：不返回任何记忆才算正确
    const score = entry.semantic_score
    return score === null || score === undefined || score < spec.expectNoStrong
  }
  if (!entry) return false
  if (spec.expectId) return entry.entry_id === spec.expectId
  if (spec.expectContains) {
    return String(entry.content ?? '').toLowerCase().includes(spec.expectContains.toLowerCase())
  }
  return false
}

/** 期望结果是否出现在 top-3（区分“没召回”和“召回但排序错”）。 */
function matchTop3(entries, spec) {
  if (spec.expectNoStrong !== undefined) return !entries.length
  return entries.some((e) => {
    if (spec.expectId) return e.entry_id === spec.expectId
    if (spec.expectContains) {
      return String(e.content ?? '').toLowerCase().includes(spec.expectContains.toLowerCase())
    }
    return false
  })
}

const report = { label, root, db: 'copy', queries: [], protocol: {}, ingest: {}, duration_ms: 0 }
const t0 = Date.now()

try {
  const init = await request('initialize', {
    protocolVersion: '2024-11-05',
    capabilities: {},
    clientInfo: { name: 'mem-regress-10', version: '0.1.0' },
  })
  if (init.error) fail(`initialize 失败: ${JSON.stringify(init.error)}`)
  notify('notifications/initialized')

  const status = await call('memory_status')
  report.embedder = status.embedder?.mode
  report.tools = (await request('tools/list', {})).result?.tools?.length ?? 0

  // ── 10 组查询 ─────────────────────────────────────────────────────────────
  for (const spec of QUERIES) {
    const res = await call('memory_search', { query: spec.q, limit: 5 })
    const entries = res.results ?? []
    const top1 = entries[0] ?? null
    report.queries.push({
      query: spec.q,
      expect: spec.expectId ?? spec.expectContains ?? `score<${spec.expectNoStrong}`,
      top1: top1?.entry_id ?? null,
      top1_score: top1?.semantic_score ?? null,
      hit: matchQuery(top1, spec),
      top3_hit: matchTop3(entries.slice(0, 3), spec),
    })
  }

  // ── 协议门场景 ────────────────────────────────────────────────────────────
  // A. 同一轮 weave 两次（同一条消息）——两版都应通过
  const a1 = await call('memory_weave', { user_message: '协议测试 A 第一轮消息' })
  const a2 = await call('memory_weave', { user_message: '协议测试 A 第一轮消息' })
  report.protocol.A_same_turn_weave = {
    first: a1.error ? a1.error.slice(0, 60) : 'ok',
    second: a2.error ? a2.error.slice(0, 60) : 'ok',
  }

  // B. 换一条新消息再 weave —— 旧版：ClassificationRequired
  const b = await call('memory_weave', { user_message: '协议测试 B 全新消息' })
  report.protocol.B_unclassified_next_turn = {
    result: b.error ? b.error.slice(0, 80) : 'ok',
    notice: (b.protocol_notice ?? '').slice(0, 80),
  }

  // C. classify(mission) 后 weave —— 旧版：SkillCheckRequired
  await call('memory_classify', { category: 'mission', note: '回归测试用 mission', gaps: [] })
  const c = await call('memory_weave', { user_message: '协议测试 C 分类为 mission 之后' })
  report.protocol.C_mission_gate = {
    result: c.error ? c.error.slice(0, 80) : 'ok',
    notice: (c.protocol_notice ?? '').slice(0, 80),
  }

  // D. classify(chat) 后 weave —— 两版都应通过，并带可注入文本
  await call('memory_classify', { category: 'chat' })
  const d = await call('memory_weave', { user_message: '协议测试 D 分类为 chat 之后' })
  report.protocol.D_after_classify = {
    result: d.error ? d.error.slice(0, 80) : 'ok',
    inject_text_chars: typeof d.text === 'string' ? d.text.length : 0,
    has_prompt_block: typeof d.prompt_block === 'string',
  }

  // ── weave 写副作用 & 角色保真 ─────────────────────────────────────────────
  // 模拟真实插件流程：pre-step 自动 weave(userText) → turn-stopping 把
  // userText + 助手回复拼成一段以 role=user 入库（旧实现）。旧版 weave 有写
  // 副作用，同一句会以两种文本形态各存一份；新版只由 turn-stopping 存一份。
  const marker = `回归标记-${label}-${Date.now()}`
  await call('memory_weave', { user_message: marker })
  await call('memory_ingest', { content: `${marker}\n\n这是助手的回复正文`, role: 'user' })
  const roleMarker = `角色标记-${label}-${Date.now()}`
  await call('memory_ingest', { content: roleMarker, role: 'assistant' })

  const counted = sqlite(`import sqlite3,json
c=sqlite3.connect(r'${dbPath.replace(/\\/g, '/')}')
rows=c.execute("select role,count(*) from dialogue_turns where content like ? group by role", ("%${marker}%",)).fetchall()
roles=c.execute("select role,count(*) from dialogue_turns where content like ? group by role", ("%${roleMarker}%",)).fetchall()
print(json.dumps({"marker_rows_by_role": rows, "role_marker_rows_by_role": roles}, ensure_ascii=False))`)
  report.ingest = counted

  const hits = report.queries.filter((q) => q.hit).length
  const hits3 = report.queries.filter((q) => q.top3_hit).length
  report.accuracy = `${hits}/${report.queries.length}`
  report.accuracy_top3 = `${hits3}/${report.queries.length}`
  report.duration_ms = Date.now() - t0

  // ── 输出 ──────────────────────────────────────────────────────────────────
  console.log(`\n[${label}] root=${root}`)
  console.log(`embedder=${report.embedder} tools=${report.tools} db=copy`)
  console.log('\n# 查询 | 期望 | top1 | 分数 | top1 | top3')
  for (const q of report.queries) {
    console.log(`- ${q.query} | ${q.expect} | ${q.top1 ?? '(空)'} | ${q.top1_score ?? '-'} | ${q.hit ? '✅' : '❌'} | ${q.top3_hit ? '✅' : '❌'}`)
  }
  console.log(`\n准确率 top1: ${report.accuracy}   top3: ${report.accuracy_top3}`)
  console.log('\n# 协议门')
  console.log('A 同轮双 weave :', JSON.stringify(report.protocol.A_same_turn_weave))
  console.log('B 未分类下一轮  :', JSON.stringify(report.protocol.B_unclassified_next_turn))
  console.log('C mission 门    :', JSON.stringify(report.protocol.C_mission_gate))
  console.log('D 分类后 weave  :', JSON.stringify(report.protocol.D_after_classify))
  console.log('\n# 入库副作用')
  console.log('weave+ingest 同一句 :', JSON.stringify(report.ingest.marker_rows_by_role))
  console.log('assistant 角色标记  :', JSON.stringify(report.ingest.role_marker_rows_by_role))
  console.log(`\n用时 ${report.duration_ms}ms`)

  const outPath = path.join(path.resolve(here, '..'), `regress-${label}.json`)
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2), 'utf8')
  console.log(`报告: ${outPath}`)

  child.kill()
  process.exit(0)
} catch (error) {
  fail(error instanceof Error ? error.message : String(error))
}

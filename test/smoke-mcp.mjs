#!/usr/bin/env node
/**
 * MCP stdio 冒烟测试：直接拉起内置后端，不经过 dsh。
 *
 *   node test/smoke-mcp.mjs
 *
 * 验证：
 *   1. initialize 握手成功
 *   2. tools/list 返回 21 个 memory_* 工具
 *   3. memory_status 可用，embedder 处于 onnx 模式
 *   4. memory_facts 能读出事实卡（至少 1 张）
 */
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const launcher = path.join(root, 'scripts', 'run-mcp.cjs')

// 无条件指向示例库：宿主 shell 可能已把 MEMORY_SKILL_DB_PATH 钉到真实库，
// 冒烟测试绝不允许读写它。要指定别的库用 MEMORY_SMOKE_DB。
process.env.MEMORY_SKILL_DB_PATH =
  process.env.MEMORY_SMOKE_DB || path.join(root, 'samples', 'sample.db')

const child = spawn(process.execPath, [launcher], {
  cwd: root,
  env: process.env,
  stdio: ['pipe', 'pipe', 'pipe'],
})

const rl = createInterface({ input: child.stdout })
const pending = new Map()
let stderrTail = ''
child.stderr.on('data', (chunk) => {
  stderrTail = (stderrTail + chunk.toString()).slice(-4000)
})

rl.on('line', (line) => {
  let message
  try {
    message = JSON.parse(line)
  } catch {
    return // 非 JSON 行（日志）忽略
  }
  const id = message.id
  if (id !== undefined && pending.has(id)) {
    pending.get(id)(message)
    pending.delete(id)
  }
})

function request(id, method, params) {
  child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${method} 超时（120s）`)), 120_000)
    pending.set(id, (message) => {
      clearTimeout(timer)
      resolve(message)
    })
  })
}

function notify(method, params) {
  child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n')
}

function fail(message) {
  console.error(`\n✗ ${message}`)
  if (stderrTail) console.error(`\n--- MCP stderr（末尾）---\n${stderrTail}`)
  child.kill()
  process.exit(1)
}

try {
  const init = await request(1, 'initialize', {
    protocolVersion: '2024-11-05',
    capabilities: {},
    clientInfo: { name: 'memory-protocol-smoke', version: '0.4.0' },
  })
  if (init.error) fail(`initialize 失败：${JSON.stringify(init.error)}`)
  notify('notifications/initialized')
  console.log('✓ initialize')

  const tools = await request(2, 'tools/list', {})
  const names = (tools.result?.tools ?? []).map((t) => t.name).sort()
  console.log(`✓ tools/list：${names.length} 个工具`)
  if (names.length !== 21) fail(`工具数量应为 21，实际 ${names.length}：${names.join(', ')}`)

  const status = await request(3, 'tools/call', { name: 'memory_status', arguments: {} })
  const statusText = status.result?.content?.[0]?.text ?? ''
  const statusJson = JSON.parse(statusText)
  console.log(`✓ memory_status：embedder=${statusJson.embedder?.mode}, db=${statusJson.config?.db_path}`)
  if (statusJson.embedder?.mode !== 'onnx') {
    fail(`embedder 不是 onnx（当前 ${statusJson.embedder?.mode}），模型路径有问题`)
  }

  const facts = await request(4, 'tools/call', { name: 'memory_facts', arguments: {} })
  const factsText = facts.result?.content?.[0]?.text ?? ''
  const factsJson = JSON.parse(factsText)
  const count = Array.isArray(factsJson) ? factsJson.length : factsJson.facts?.length
  console.log(`✓ memory_facts：${count} 张事实卡`)
  if (!count) fail('事实卡为空，先跑 python samples/make_sample_db.py --force')

  console.log('\n冒烟测试通过。')
  child.kill()
  process.exit(0)
} catch (error) {
  fail(error instanceof Error ? error.message : String(error))
}

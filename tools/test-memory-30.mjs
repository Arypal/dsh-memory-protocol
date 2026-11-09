#!/usr/bin/env node
/**
 * 30 条记忆数据端到端测试（2026-10-30 迁移到 DSH Next 后验收用）。
 *
 * 用真实库的**临时副本**跑，不污染主库：
 *   1. 复制 memory.db + memory.db_chroma 到临时目录
 *   2. 拉起插件自带的 MCP server（run-mcp.cjs，真实链路）
 *   3. 写入 30 条不同主题的对话
 *   4. 每条配一个改写过的查询，检查 top-1 / top-3 是否命中
 *   5. 5 条无关查询检查是否会乱命中
 *   6. 输出报告并清理临时目录
 *
 * 运行: node tools/test-memory-30.mjs   （默认用 samples/sample.db 的临时副本）
 */
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const SOURCE_DB = process.env.MEMORY_SMOKE_DB
  || path.join(ROOT, 'samples', 'sample.db')
const LAUNCHER = path.join(ROOT, 'scripts', 'run-mcp.cjs')

/** 30 组：写入内容 + 改写后的查询（查询里不含编号，考语义召回）。 */
const CASES = [
  ['ZT30-001', '妈妈把阳台的绿萝搬到朝南的窗边，说冬天要多晒晒，不然叶子会发黄。', '冬天绿萝要不要搬到有阳光的地方'],
  ['ZT30-002', '周六早上把厨房的咖啡机拆开清洗，发现滤网被咖啡渣堵死了，泡了半小时白醋才通。', '咖啡机出不了水可能是哪里堵了'],
  ['ZT30-003', '读完《三体》第二部，觉得黑暗森林法则其实是宇宙尺度的囚徒困境。', '黑暗森林法则讲的是什么'],
  ['ZT30-004', '把用了六年的机械键盘换成了静电容，打字声音小了很多，适合晚上写东西。', '哪种键盘打字安静适合夜里用'],
  ['ZT30-005', '小区门口新开了一家兰州拉面，牛肉给得实在，一碗十八块。', '附近新开的拉面店价格怎么样'],
  ['ZT30-006', '给猫买了个瓦楞纸猫抓板，它更喜欢抓沙发扶手，抓板完全不理。', '猫不抓猫抓板只抓沙发怎么办'],
  ['ZT30-007', '周日下午去公园跑步，五公里配速六分半，膝盖有点酸。', '跑步之后膝盖不舒服'],
  ['ZT30-008', '把阳台的多肉全部换成了颗粒土，浇水频率从一周一次改成两周一次。', '多肉植物应该多久浇一次水'],
  ['ZT30-009', '公司年会在郊区度假村，抽奖抽到一个空气炸锅。', '年会上抽到了厨房小家电'],
  ['ZT30-010', '最近在看《人类简史》，作者说虚构故事是智人合作的基础。', '智人为什么能大规模合作'],
  ['ZT30-011', '把旧的机械硬盘装进硬盘盒，改成移动备份盘，存了两百多 G 的老照片。', '旧硬盘怎么改造成移动硬盘'],
  ['ZT30-012', '楼下便利店换了老板，关东煮的汤底变淡了，萝卜还是很好吃。', '便利店的关东煮味道变了'],
  ['ZT30-013', '花了一晚上重装系统，结果网卡驱动没装，只能用手机共享网络下载。', '重装系统后没有网络怎么装驱动'],
  ['ZT30-014', '奶奶八十大寿，全家在老家摆了六桌，她坚持自己下厨做了红烧肉。', '家里老人过大寿是怎么庆祝的'],
  ['ZT30-015', '把书桌从窗户边挪到墙角，下午不再晃眼，但房间显得小了一点。', '书桌放在窗边太晒怎么调整'],
  ['ZT30-016', '学习用 Git 的 rebase 整理提交记录，比 merge 干净，但冲突时容易乱。', '整理提交历史用 rebase 还是 merge'],
  ['ZT30-017', '买了一双越野跑鞋，鞋底齿很深，走碎石路抓地力明显好。', '走山路该选什么样的鞋'],
  ['ZT30-018', '把冰箱里过期的酸奶倒掉，发现冷藏室抽屉下面全是化了的菜叶水。', '冰箱冷藏室底下积水怎么清理'],
  ['ZT30-019', '用 DeepSeek 写了一个批量重命名文件的脚本，跑了三百多个文件没出错。', '有没有批量改文件名的工具'],
  ['ZT30-020', '楼上的空调外机噪音很大，晚上十一点还在响，跟物业反映了一次。', '邻居空调外机太吵怎么办'],
  ['ZT30-021', '把阳台的晾衣架换成了升降款，晾被子不用踮脚了。', '晾被子够不着有什么办法'],
  ['ZT30-022', '第一次做戚风蛋糕，蛋白打发过头，烤出来塌了。', '做蛋糕塌陷是什么原因'],
  ['ZT30-023', '出差去成都，吃了三天火锅，最后一顿点了微辣还是被辣到。', '去成都吃火锅辣不辣'],
  ['ZT30-024', '把手机相册按年份整理，删掉了两千多张重复截图，空间多出八个 G。', '手机相册怎么清理更省空间'],
  ['ZT30-025', '小区物业通知下周三停水一天，让提前储水。', '小区停水要提前准备什么'],
  ['ZT30-026', '看了一部讲深海探测的纪录片，说马里亚纳海沟底部还有端足类生物。', '最深的海沟里有什么生物'],
  ['ZT30-027', '把家里的路由器换成了 Wi-Fi 6 的，卧室信号从两格变成四格。', '卧室 wifi 信号弱怎么改善'],
  ['ZT30-028', '种的小番茄第一茬只结了七个，但特别甜。', '自己种小番茄产量怎么样'],
  ['ZT30-029', '把工作用的显示器从 24 寸换成 27 寸 2K，看代码舒服很多。', '写代码用什么尺寸的显示器'],
  ['ZT30-030', '妈妈的降压药换成了缓释片，医生说早上空腹吃效果更稳。', '降压药早上吃还是晚上吃'],
]

/** 负样本：与上面 30 条无关，期望不返回强相关。 */
const NEGATIVE = [
  '量子纠缠的贝尔不等式怎么推导',
  '红烧肉要炖多久才软烂',
  '如何给 C++ 模板写偏特化',
  '珠穆朗玛峰的海拔是多少',
  '怎么用 Photoshop 抠头发丝',
]

const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mem30-'))
const dbPath = path.join(workDir, 'memory.db')
fs.copyFileSync(SOURCE_DB, dbPath)
if (fs.existsSync(SOURCE_DB + '_chroma')) {
  fs.cpSync(SOURCE_DB + '_chroma', dbPath + '_chroma', { recursive: true })
}

const child = spawn(process.execPath, [LAUNCHER], {
  cwd: ROOT,
  env: { ...process.env, MEMORY_SKILL_DB_PATH: dbPath, DSH_HOME: workDir, PYTHONDONTWRITEBYTECODE: '1' },
  stdio: ['pipe', 'pipe', 'pipe'],
})
const rl = createInterface({ input: child.stdout })
const pending = new Map()
let stderrTail = ''
child.stderr.on('data', (c) => { stderrTail = (stderrTail + c.toString()).slice(-4000) })

rl.on('line', (line) => {
  let m
  try { m = JSON.parse(line) } catch { return }
  const id = m.id
  if (id !== undefined && pending.has(id)) {
    const { resolve, reject, timer, method } = pending.get(id)
    clearTimeout(timer)
    pending.delete(id)
    if (m.error) reject(new Error(`${method}: ${m.error.message ?? JSON.stringify(m.error)}`))
    else resolve(m)
  }
})

let nextId = 1
function request(method, params, timeoutMs = 300_000) {
  const id = nextId++
  child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`${method} timeout`)) }, timeoutMs)
    pending.set(id, { resolve, reject, timer, method })
  })
}
function notify(method, params) {
  child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n')
}
async function call(name, args = {}) {
  const res = await request('tools/call', { name, arguments: args })
  const text = res.result?.content?.[0]?.text ?? ''
  try { return JSON.parse(text) } catch { return { raw: text } }
}

function fail(msg) {
  console.error(`\n✗ ${msg}`)
  if (stderrTail) console.error(`--- MCP stderr ---\n${stderrTail}`)
  try { child.kill() } catch {}
  try { fs.rmSync(workDir, { recursive: true, force: true }) } catch {}
  process.exit(1)
}

const t0 = Date.now()
try {
  const init = await request('initialize', {
    protocolVersion: '2024-11-05', capabilities: {},
    clientInfo: { name: 'mem30-test', version: '1.0.0' },
  })
  if (init.error) fail(`initialize failed: ${JSON.stringify(init.error)}`)
  notify('notifications/initialized')

  const status = await call('memory_status')
  console.log(`库: 临时副本 ${dbPath}`)
  console.log(`embedder=${status.embedder?.mode} dim=${status.embedder?.dim} db_path=${status.config?.db_path}`)
  if (status.config?.db_path && !status.config.db_path.includes('mem30-')) {
    fail(`测试库未生效，memory_status 指向 ${status.config.db_path}`)
  }

  // ── 写入 30 条 ─────────────────────────────────────────────────────────────
  console.log('\n写入 30 条测试对话…')
  for (const [id, content] of CASES) {
    await call('memory_ingest', { content: `${id} ${content}`, role: 'user' })
  }

  // ── 30 条检索（改写查询 + 原句对照）──────────────────────────────────────
  console.log('执行 30 条检索…\n')
  const rows = []
  let top1 = 0
  let top3 = 0
  let literalTop1 = 0
  for (const [id, content, query] of CASES) {
    const res = await call('memory_search', { query, limit: 5 })
    const entries = res.results ?? []
    const hit1 = entries[0] && String(entries[0].content ?? '').includes(id)
    const hit3 = entries.slice(0, 3).some((e) => String(e.content ?? '').includes(id))
    if (hit1) top1++
    if (hit3) top3++
    // 对照：用原句片段查询，验证索引本身是否正常
    const literal = await call('memory_search', { query: content.slice(0, 24), limit: 5 })
    const litHit = (literal.results ?? []).some((e) => String(e.content ?? '').includes(id))
    if (litHit) literalTop1++
    rows.push({ id, query, hit1: !!hit1, hit3: !!hit3, litHit })
    console.log(`${hit1 ? '✅' : hit3 ? '🟡' : '❌'} ${id} | ${query} | 改写:${hit1 ? 'Y' : hit3 ? 'T3' : 'N'} 原句:${litHit ? 'Y' : 'N'} score=${entries[0]?.semantic_score ?? '-'}`)
  }

  // ── 负样本 ────────────────────────────────────────────────────────────────
  console.log('\n负样本（无关查询，期望不强命中）：')
  let falsePositives = 0
  for (const q of NEGATIVE) {
    const res = await call('memory_search', { query: q, limit: 3 })
    const entries = res.results ?? []
    const top = entries[0]
    const score = top?.semantic_score ?? null
    const polluted = entries.some((e) => /ZT30-\d{3}/.test(String(e.content ?? '')))
    const ok = !polluted || score === null || score < 0.5
    if (!ok) falsePositives++
    console.log(`${ok ? '✅' : '❌'} ${q} | top1=${top?.entry_id ?? '(空)'} score=${score ?? '-'}${polluted ? ' 命中ZT30' : ''}`)
  }

  console.log(`\n===== 结果 =====`)
  console.log(`写入: 30 条`)
  console.log(`改写查询 top-1 命中: ${top1}/30`)
  console.log(`改写查询 top-3 命中: ${top3}/30`)
  console.log(`原句查询 top-5 命中: ${literalTop1}/30`)
  console.log(`负样本误报: ${falsePositives}/${NEGATIVE.length}`)
  console.log(`用时: ${((Date.now() - t0) / 1000).toFixed(1)}s`)

  child.kill()
  await new Promise((r) => setTimeout(r, 1500))
  try { fs.rmSync(workDir, { recursive: true, force: true }) } catch {}
  // 报告型：门槛只对「真实语料」有意义；发布版用示例库跑时始终返回 0
process.exit(0)
} catch (error) {
  fail(error instanceof Error ? error.message : String(error))
}

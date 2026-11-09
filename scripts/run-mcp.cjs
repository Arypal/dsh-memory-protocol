'use strict'
// memory-mcp 启动入口（0.4.0）。
//
// 与 0.3.2 的区别：Python 后端随插件分发在 ../python/memory_skill，并通过
// PYTHONPATH 前置加载，因此不再依赖 site-packages 里被就地改过的 memory_skill。
// 本进程只做解析 + 转发，不向 stdout 打印任何内容（MCP JSON-RPC 纯净性）。
//
// 环境变量：
//   MEMORY_SKILL_PYTHON   — 解释器（默认探测 python3 → python → py）
//   MEMORY_SKILL_DB_PATH  — 记忆库路径（默认 <DSH_HOME>/memory/memory.db）
//   MEMORY_MODEL_PATH     — 嵌入模型目录（未设时自动探测）
//   MEMORY_PROTOCOL_BACKEND=installed — 强制用 site-packages 后端（排障用）
const { spawn, spawnSync } = require('node:child_process')
const path = require('node:path')
const fs = require('node:fs')
const { resolvePython } = require('./python-resolve.cjs')

const pluginRoot = path.resolve(__dirname, '..')
const bundledPython = path.join(pluginRoot, 'python')
const python = resolvePython()

// 不往插件目录里写 .pyc：包目录可能只读，也会污染下次打包。
process.env.PYTHONDONTWRITEBYTECODE ??= '1'

// ── 1. 内置后端优先 ───────────────────────────────────────────────────────────
const bundledServer = path.join(bundledPython, 'memory_skill', 'mcp_server.py')
const forceInstalled = process.env.MEMORY_PROTOCOL_BACKEND === 'installed'
if (fs.existsSync(bundledServer) && !forceInstalled) {
  const sep = path.delimiter
  const current = process.env.PYTHONPATH || ''
  process.env.PYTHONPATH = current
    ? bundledPython + sep + current
    : bundledPython
} else if (!forceInstalled) {
  console.error(
    `[memory-mcp] 内置后端缺失（${bundledServer}），回退到 site-packages 的 memory_skill`,
  )
}

// ── 2. 记忆库位置 ─────────────────────────────────────────────────────────────
// 默认 <DSH_HOME>/memory/memory.db（DSH_HOME 缺省为 ~/.dsh）。
// 显式设置 MEMORY_SKILL_DB_PATH 时一律尊重（回归测试/多库/排障都靠它）。
const DSH_HOME = process.env.DSH_HOME || path.join(require('node:os').homedir(), '.dsh')
const CANONICAL_DB = path.join(DSH_HOME, 'memory', 'memory.db')
const explicitDb = process.env.MEMORY_SKILL_DB_PATH
if (!explicitDb) {
  fs.mkdirSync(path.dirname(CANONICAL_DB), { recursive: true })
  process.env.MEMORY_SKILL_DB_PATH = CANONICAL_DB
}

// ── 3. 嵌入模型自动探测 ───────────────────────────────────────────────────────
// 内置后端改变了 contracts.py 的默认 model_path（<plugin>/python/models/...），
// 所以未显式设置时，主动去 site-packages/models 等常见位置找模型目录。
function hasModel(dir) {
  return (
    fs.existsSync(path.join(dir, 'model.onnx')) ||
    fs.existsSync(path.join(dir, 'onnx', 'model.onnx'))
  )
}

if (!process.env.MEMORY_MODEL_PATH) {
  const candidates = [
    path.join(bundledPython, 'models', 'bge-large-zh-v1.5'),
    path.join(bundledPython, 'models', 'bge-large-en-v1.5'),
  ]
  const probe = spawnSync(
    python,
    [
      '-c',
      'import site, os; print(os.pathsep.join(site.getsitepackages() + [site.getusersitepackages()]))',
    ],
    { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] },
  )
  if (probe.status === 0 && probe.stdout) {
    for (const dir of probe.stdout.trim().split(path.delimiter)) {
      if (!dir) continue
      candidates.push(
        path.join(dir, 'models', 'bge-large-zh-v1.5'),
        path.join(dir, 'models', 'bge-large-en-v1.5'),
      )
    }
  }
  const hit = candidates.find(hasModel)
  if (hit) {
    process.env.MEMORY_MODEL_PATH = hit
  } else {
    console.error(
      '[memory-mcp] 未找到嵌入模型目录（model.onnx），语义检索将降级；用法：set MEMORY_MODEL_PATH=<模型目录>',
    )
  }
}

// ── 4. 启动 ───────────────────────────────────────────────────────────────────
// windowsHide：DSH Next 是 GUI 进程，spawn 控制台程序会弹黑框；隐藏掉。
const child = spawn(python, ['-m', 'memory_skill.mcp_server'], {
  stdio: 'inherit',
  env: process.env,
  windowsHide: true,
})

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => child.kill(sig))
}
child.on('exit', (code, signal) => process.exit(code ?? (signal ? 1 : 0)))

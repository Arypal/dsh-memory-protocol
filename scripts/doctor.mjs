#!/usr/bin/env node
/**
 * memory-protocol 0.4.0 自检（doctor）。
 *
 * 0.4.0 起 Python 后端随插件分发（../python/memory_skill），不再需要
 * `pip install memory-skill`；本脚本只做只读检查，给出可执行的修复命令，
 * 永远 exit 0（fail-open，绝不阻塞 dsh 启动）。
 *
 * 检查项：
 *   1. Python 解释器
 *   2. 内置后端能否被 PYTHONPATH 优先加载
 *   3. 运行时依赖（mcp / chromadb / onnxruntime / numpy / jieba）
 *   4. 嵌入模型目录（model.onnx）
 *   5. 记忆库路径与表结构
 *   6. IMPORTANCE_API_KEY（可选，distill/主动学习用）
 */
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { resolvePython } from './python-resolve.cjs'

const log = (...args) => console.error('[memory-doctor]', ...args)
const pluginRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const bundledPython = path.join(pluginRoot, 'python')
const bundledServer = path.join(bundledPython, 'memory_skill', 'mcp_server.py')
const PY = resolvePython()

let failures = 0
const ok = (msg) => log('OK   ', msg)
const bad = (msg) => {
  failures += 1
  log('FAIL ', msg)
}
const warn = (msg) => log('WARN ', msg)

function run(args, extraEnv = {}) {
  return spawnSync(PY, args, {
    encoding: 'utf8',
    env: { PYTHONDONTWRITEBYTECODE: '1', ...process.env, ...extraEnv },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
}

function checkPython() {
  const r = run(['--version'])
  const version = (r.stdout || r.stderr || '').trim()
  if (r.status === 0) ok(`python: ${version}（${PY}）`)
  else bad(`python 不可用：${PY}；设 MEMORY_SKILL_PYTHON 指向真实 python.exe`)
}

function checkBackend() {
  if (!existsSync(bundledServer)) {
    bad(`内置后端缺失：${bundledServer}`)
    return
  }
  const r = run(['-c', 'import memory_skill; print(memory_skill.__file__)'], {
    PYTHONPATH: bundledPython,
  })
  const where = (r.stdout || '').trim()
  if (r.status !== 0) {
    bad(`内置后端 import 失败：${(r.stderr || '').trim().split('\n').slice(-3).join(' | ')}`)
    return
  }
  if (where.startsWith(bundledPython)) ok(`内置后端：${where}`)
  else warn(`import 命中的不是内置后端：${where}（查 PYTHONPATH / MEMORY_PROTOCOL_BACKEND）`)
}

function checkDeps() {
  const r = run(
    [
      '-c',
      'import importlib.util,sys; mods=["mcp","chromadb","onnxruntime","numpy","jieba"]; ' +
        'missing=[m for m in mods if importlib.util.find_spec(m) is None]; ' +
        'print(",".join(missing))',
    ],
    { PYTHONPATH: bundledPython },
  )
  const missing = (r.stdout || '').trim()
  if (r.status !== 0) {
    warn('依赖检查失败（跳过）')
    return
  }
  if (!missing) {
    ok('运行时依赖齐全（mcp/chromadb/onnxruntime/numpy/jieba）')
    return
  }
  bad(`缺少依赖：${missing}`)
  log(`     修复：${PY} -m pip install ${missing.split(',').join(' ')}`)
}

function hasModel(dir) {
  return (
    existsSync(path.join(dir, 'model.onnx')) ||
    existsSync(path.join(dir, 'onnx', 'model.onnx'))
  )
}

function resolveModelDir() {
  if (process.env.MEMORY_MODEL_PATH) return process.env.MEMORY_MODEL_PATH
  const candidates = [
    path.join(bundledPython, 'models', 'bge-large-zh-v1.5'),
    path.join(bundledPython, 'models', 'bge-large-en-v1.5'),
  ]
  const probe = run(['-c', 'import site,os;print(os.pathsep.join(site.getsitepackages()+[site.getusersitepackages()]))'])
  if (probe.status === 0 && probe.stdout) {
    for (const dir of probe.stdout.trim().split(path.delimiter)) {
      if (!dir) continue
      candidates.push(
        path.join(dir, 'models', 'bge-large-zh-v1.5'),
        path.join(dir, 'models', 'bge-large-en-v1.5'),
      )
    }
  }
  return candidates.find(hasModel)
}

function checkModel() {
  const dir = resolveModelDir()
  if (!dir) {
    bad('未找到嵌入模型目录（model.onnx）')
    log('     修复：set MEMORY_MODEL_PATH=<含 model.onnx 的目录>，或从 HF 下载 bge-large-zh-v1.5')
    return
  }
  ok(`嵌入模型：${dir}`)
}

function checkDb() {
  const dbPath = process.env.MEMORY_SKILL_DB_PATH || 'D:\\DSH\\memory.db'
  if (!existsSync(dbPath)) {
    warn(`记忆库不存在（首次启动会自动建）：${dbPath}`)
    return
  }
  const r = run(
    [
      '-c',
      'import sqlite3,sys; con=sqlite3.connect(sys.argv[1]); ' +
        'print(con.execute("select count(*) from dialogue_turns").fetchone()[0], ' +
        'con.execute("select count(*) from facts").fetchone()[0])',
      dbPath,
    ],
    { PYTHONPATH: bundledPython },
  )
  const [turns, facts] = (r.stdout || '').trim().split(/\s+/)
  if (r.status === 0 && turns !== undefined) ok(`记忆库：${dbPath}（对话 ${turns} 条 / 事实卡 ${facts} 张）`)
  else warn(`记忆库可读但结构异常：${(r.stderr || '').trim().split('\n').slice(-2).join(' | ')}`)
}

function checkImportanceKey() {
  const r = run([
    '-c',
    'import os;from pathlib import Path;from dotenv import load_dotenv;' +
      'load_dotenv(Path.home()/".config"/"memory-skill"/".env",override=False);' +
      'print(bool(os.environ.get("IMPORTANCE_API_KEY","")))',
  ])
  if ((r.stdout || '').trim() === 'True') ok('IMPORTANCE_API_KEY 已配置（distill/主动学习可用）')
  else warn('IMPORTANCE_API_KEY 未配置（distill/主动学习不可用；记忆核心不受影响）')
}

log(`plugin: ${pluginRoot}`)
checkPython()
checkBackend()
checkDeps()
checkModel()
checkDb()
checkImportanceKey()

if (failures === 0) log('自检完成：可用')
else log(`自检完成：${failures} 项失败，按上面的修复命令处理后重启 dsh`)
process.exit(0)

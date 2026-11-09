#!/usr/bin/env node
/**
 * 0.4.0：Python 后端已随插件分发（../python/memory_skill），不再需要
 * `pip install memory-skill`。原先的「自动装依赖 + 下载模型」降级为只读自检：
 * 输出缺失项和可执行的修复命令，永远 exit 0。
 *
 * 手动执行：node node_modules/dsh-memory-protocol/scripts/doctor.mjs
 * 关闭自检：MEMORY_SKIP_BOOTSTRAP=1
 */
if (process.env.MEMORY_SKIP_BOOTSTRAP === '1') {
  console.error('[memory-bootstrap] skipped (MEMORY_SKIP_BOOTSTRAP=1)')
  process.exit(0)
}
await import('./doctor.mjs')

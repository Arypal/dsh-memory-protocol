# 打包与发布说明

## 这个仓库包含什么

```
dsh-memory-protocol/            ← 直接作为 DSH 插件安装（file: 依赖）
├─ lib/                        协议层（pre-step / turn-stopping / 软提醒门）
├─ client/                     可选前端组件（记忆开关按钮）
├─ cordis.patch.yml            bundle 补丁
├─ python/memory_skill/        内置后端（43 个模块，无需 pip install memory-skill）
├─ scripts/                    run-mcp.cjs / python-resolve.cjs / doctor.mjs / bootstrap-memory.mjs
├─ test/                       smoke-mcp.mjs / regress-10.mjs
├─ tools/                      backup · distill · purge_secrets · encrypt_migrate · rekey · test-memory-30
├─ samples/                    make_sample_db.py（只含虚构数据）+ 说明
├─ docs/                       ARCHITECTURE.md + OPTIONAL_MODULES.md
├─ README.md / README.en.md
├─ requirements.txt
├─ install.ps1                 一键安装（Windows）
├─ LICENSE                     MIT
└─ .gitignore
```

## 这个仓库**不**包含

| 排除项 | 原因 |
|---|---|
| `memory.db` / `memory.db_chroma/` | 真实记忆 |
| `master.key` | 解密密钥（与数据同机才有意义） |
| `backups/`、`*.mbk` | 真实记忆的加密快照 |
| `paused.json`、`runtime/`、`*.log` | 运行时状态 |
| 任何 `.dsh` 路径与主机名 | 已在代码里改为环境变量推导 |
| 嵌入模型（~1.2 GB） | 体积过大，README 给下载与配置方式 |

`.gitignore` 与 `samples/.gitignore` 已双重覆盖；`git status` 里不应出现上述任何文件。

## 发布前检查

```powershell
cd <本仓库>
git init
git add .
git status --short            # 逐条确认没有 db / key / log / 备份
git commit -m "feat: dsh-memory-protocol 0.4.1"

# 冒烟
pip install -r requirements.txt
python samples/make_sample_db.py --force
node scripts/doctor.mjs
node test/smoke-mcp.mjs       # 期望 21 工具 / embedder=onnx / 4 张卡
node tools/test-memory-30.mjs # 基准报告（示例库语料少，命中率不代表线上）

# 清理本地生成物后再提交
Remove-Item samples/sample.db, samples/sample.db_chroma, samples/master.key -Recurse -Force -ErrorAction SilentlyContinue
```

## 打 tag

```bash
git tag -a v0.4.1 -m "dsh-memory-protocol 0.4.1"
git push origin main --tags
```

## 校验"下载即可用"的三条硬指标

1. `pip install -r requirements.txt` 无缺失依赖；
2. 设置 `MEMORY_MODEL_PATH` 后 `node test/smoke-mcp.mjs` 全绿；
3. `dsh plugin --profile <profile> add file:<repo>` + 重启宿主后，`memory_status` 显示 `status: healthy`。

缺模型时第 2 条会降级为 BM25（`embedder.mode=fallback`），不算失败，但要在 README 告知用户。

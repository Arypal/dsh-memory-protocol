#Requires -Version 5.1
<#
.SYNOPSIS
    dsh-memory-protocol 一键安装（Windows / PowerShell）

.DESCRIPTION
    1) 校验 Python 与依赖
    2) 校验嵌入模型目录
    3) 用 dsh 把本仓库作为 file: 插件装进指定 profile
    4) 生成示例库并跑自检

.PARAMETER Profile
    DSH profile 名。宿主实际跑哪个就装哪个（默认 desktop）。

.PARAMETER ModelPath
    嵌入模型目录（含 model.onnx）。留空则读 MEMORY_MODEL_PATH，再留空则自动探测。

.PARAMETER SkipPlugin
    只装依赖与示例库，不安装插件（用于先手动验证后端）。

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Profile desktop -ModelPath "D:\models\bge-large-zh-v1.5"
#>
param(
    [string]$Profile = 'desktop',
    [string]$ModelPath = '',
    [switch]$SkipPlugin
)

$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
Push-Location $repo
try {
    Write-Host '=== 1/5 Python ===' -ForegroundColor Cyan
    $py = $null
    foreach ($cand in @('python', 'python3', 'py')) {
        try {
            $v = & $cand --version 2>$null
            if ($LASTEXITCODE -eq 0 -and $v) { $py = $cand; Write-Host "  使用 $cand ($v)"; break }
        } catch { }
    }
    if (-not $py) { throw '未找到 Python（需要 3.10+，建议 3.12/3.13）' }

    Write-Host '=== 2/5 依赖 ===' -ForegroundColor Cyan
    & $py -m pip install --upgrade -r (Join-Path $repo 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'pip install 失败' }

    Write-Host '=== 3/5 嵌入模型 ===' -ForegroundColor Cyan
    if (-not $ModelPath) { $ModelPath = $env:MEMORY_MODEL_PATH }
    if ($ModelPath) {
        if (-not (Test-Path (Join-Path $ModelPath 'model.onnx'))) {
            throw "模型目录里没有 model.onnx：$ModelPath"
        }
        $env:MEMORY_MODEL_PATH = $ModelPath
        Write-Host "  MEMORY_MODEL_PATH=$ModelPath"
    } else {
        Write-Host '  未指定模型：语义检索会降级为 BM25（仍可用）' -ForegroundColor Yellow
        Write-Host '  下载 bge-large-zh-v1.5 后设置 MEMORY_MODEL_PATH 再重启宿主' -ForegroundColor Yellow
    }

    Write-Host '=== 4/5 自检 ===' -ForegroundColor Cyan
    & $py (Join-Path $repo 'samples\make_sample_db.py') --force
    node (Join-Path $repo 'scripts\doctor.mjs')

    Write-Host '=== 5/5 安装插件 ===' -ForegroundColor Cyan
    if ($SkipPlugin) {
        Write-Host '  -SkipPlugin：跳过插件安装'
    } else {
        $dsh = Get-Command dsh -ErrorAction SilentlyContinue
        if (-not $dsh) {
            Write-Host '  未找到 dsh 命令。手动安装：' -ForegroundColor Yellow
            Write-Host "    dsh plugin --profile $Profile add file:$repo" -ForegroundColor Yellow
        } else {
            & dsh plugin --profile $Profile add "file:$repo"
            if ($LASTEXITCODE -ne 0) { throw 'dsh plugin add 失败' }
            Write-Host "  已安装到 profile: $Profile —— 重启宿主后生效" -ForegroundColor Green
        }
    }

    Write-Host ''
    Write-Host '完成。验证顺序：' -ForegroundColor Green
    Write-Host '  node scripts/doctor.mjs'
    Write-Host '  node test/smoke-mcp.mjs'
    Write-Host '  node tools/test-memory-30.mjs'
}
finally {
    Pop-Location
}

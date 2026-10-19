<#
.SYNOPSIS
    一键复现：环境自检 -> 数据准备 -> 单模态基线 -> 四种融合消融 -> 拆解 -> 错误分析。

.DESCRIPTION
    在仓库根目录或任意位置执行均可（脚本会自己定位项目目录）。
    数据不在仓库里，运行前请先按 docs/DATASET.md 放好 CUB_200_subset.zip。

.EXAMPLE
    # 完整跑一遍（约 1 小时，取决于显卡）
    .\scripts\run_all.ps1

.EXAMPLE
    # 先花几分钟验证流程通不通：每个实验只跑 2 个 epoch
    .\scripts\run_all.ps1 -Epochs 2 -SkipZeroshot

.EXAMPLE
    # 只跑消融那 6 组
    .\scripts\run_all.ps1 -Only ablation
#>
param(
    [int]$Epochs = 0,               # >0 时覆盖所有训练的 epoch 数（快速验证用）
    [string[]]$Only = @(),          # 只跑指定步骤：env/data/image/text/ablation/inspect/error/zeroshot
    [switch]$SkipZeroshot           # 跳过 CLIP 零样本（需要额外下载 600 MB 权重）
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$project = $repoRoot

if (-not (Test-Path (Join-Path $project "src\build_annotations.py"))) {
    throw "当前目录不是仓库根（找不到 src\build_annotations.py）：$project"
}

# 优先用项目自带的虚拟环境
$python = "python"
$venvPython = Join-Path $project ".venv\Scripts\python.exe"
if (Test-Path $venvPython) { $python = $venvPython }

Set-Location $project

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)

    if ($Only.Count -gt 0 -and ($Only -notcontains $Name)) {
        Write-Host "`n[skip] $Name" -ForegroundColor DarkGray
        return
    }
    if ($Name -eq "zeroshot" -and $SkipZeroshot) {
        Write-Host "`n[skip] zeroshot (-SkipZeroshot)" -ForegroundColor DarkGray
        return
    }

    $argList = @()
    if ($Arguments) { $argList += $Arguments }

    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host ">>> $Name : $python $($argList -join ' ')" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    & $python @argList
    if ($LASTEXITCODE -ne 0) {
        throw "步骤 $Name 失败（exit code $LASTEXITCODE）"
    }
}

$epochArgs = @()
if ($Epochs -gt 0) { $epochArgs = @("--epochs", "$Epochs") }

Write-Host "repo    : $repoRoot"
Write-Host "project : $project"
Write-Host "python  : $python"
if ($Epochs -gt 0) { Write-Host "epochs  : $Epochs (override)" -ForegroundColor Yellow }

Invoke-Step "env"      @("src/check_env.py")
Invoke-Step "data"     @("src/build_annotations.py")
Invoke-Step "image"    (@("src/train_image_only.py") + $epochArgs)
Invoke-Step "text"     (@("src/train_text_only.py") + $epochArgs)
Invoke-Step "ablation" (@("src/ablation.py") + $epochArgs)
Invoke-Step "inspect"  @("src/inspect_features.py")
Invoke-Step "error"    @("src/error_analysis.py",
                         "--pred", "outputs/fusion_attention/test_predictions.npz",
                         "--viz_ckpt", "outputs/fusion_attention/checkpoints/best.pt")
Invoke-Step "zeroshot" @("src/zeroshot_clip.py")

Write-Host "`n全部完成。结果在：$project\outputs" -ForegroundColor Green
Write-Host "  对比表      : outputs\ablation\results.md"
Write-Host "  错误分析    : outputs\error_analysis\"
Write-Host "  特征可视化  : outputs\inspect\"

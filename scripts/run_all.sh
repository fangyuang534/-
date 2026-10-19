#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 一键复现：环境自检 -> 数据准备 -> 单模态基线 -> 四种融合消融 -> 拆解 -> 错误分析
#
# 用法：
#   bash scripts/run_all.sh                  # 完整跑一遍（约 1 小时，取决于显卡）
#   bash scripts/run_all.sh --epochs 2       # 快速验证流程
#   bash scripts/run_all.sh --only ablation  # 只跑消融
#   bash scripts/run_all.sh --skip-zeroshot  # 跳过 CLIP 零样本（需额外下载 600 MB）
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${REPO_ROOT}"          # 仓库根就是项目根

if [[ ! -f "${PROJECT}/src/build_annotations.py" ]]; then
    echo "当前目录不是仓库根（找不到 src/build_annotations.py）：${PROJECT}" >&2
    exit 1
fi

# 优先用项目自带的虚拟环境
PYTHON="python3"
if [[ -x "${PROJECT}/.venv/bin/python" ]]; then
    PYTHON="${PROJECT}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
fi

cd "${PROJECT}"

EPOCH_ARGS=()
ONLY=""
SKIP_ZEROSHOT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)        EPOCH_ARGS=(--epochs "$2"); shift 2 ;;
        --only)          ONLY="$2"; shift 2 ;;
        --skip-zeroshot) SKIP_ZEROSHOT=1; shift ;;
        -h|--help)       sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "未知参数：$1" >&2; exit 2 ;;
    esac
done

run_step() {
    local name="$1"; shift
    if [[ -n "${ONLY}" && "${ONLY}" != "${name}" ]]; then
        echo -e "\n[skip] ${name}"
        return 0
    fi
    if [[ "${name}" == "zeroshot" && "${SKIP_ZEROSHOT}" == "1" ]]; then
        echo -e "\n[skip] zeroshot (--skip-zeroshot)"
        return 0
    fi
    echo ""
    echo "============================================================"
    echo ">>> ${name} : ${PYTHON} $*"
    echo "============================================================"
    "${PYTHON}" "$@"
}

echo "repo    : ${REPO_ROOT}"
echo "project : ${PROJECT}"
echo "python  : ${PYTHON}"

run_step env      src/check_env.py
run_step data     src/build_annotations.py
# 注意 ${ARR[@]+"${ARR[@]}"} 这个写法：macOS 自带 bash 3.2 在 set -u 下
# 直接展开空数组会报 "unbound variable"，这样写才跨版本安全。
run_step image    src/train_image_only.py ${EPOCH_ARGS[@]+"${EPOCH_ARGS[@]}"}
run_step text     src/train_text_only.py  ${EPOCH_ARGS[@]+"${EPOCH_ARGS[@]}"}
run_step ablation src/ablation.py         ${EPOCH_ARGS[@]+"${EPOCH_ARGS[@]}"}
run_step inspect  src/inspect_features.py
run_step error    src/error_analysis.py \
                  --pred outputs/fusion_attention/test_predictions.npz \
                  --viz_ckpt outputs/fusion_attention/checkpoints/best.pt
run_step zeroshot src/zeroshot_clip.py

echo ""
echo "全部完成。结果在：${PROJECT}/outputs"
echo "  对比表      : outputs/ablation/results.md"
echo "  错误分析    : outputs/error_analysis/"
echo "  特征可视化  : outputs/inspect/"

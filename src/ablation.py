"""ablation.py —— 一键跑完整消融实验，并生成结果表（CSV + Markdown）

跑什么
-----
    image_only : 只看图（ResNet18）            —— 基线 A
    text_only  : 只看文本（BERT）              —— 基线 B
    early      : 早融合（concat + MLP）
    late       : 晚融合（两个头 + 可学习权重）
    attention  : 跨模态注意力（图像 token 作 Q）
    gated      : 门控融合（逐样本决定更信谁）

用法
----
    python src/ablation.py                          # 全部 6 组，默认 8 epoch
    python src/ablation.py --epochs 3                # 快速验证流程能跑通
    python src/ablation.py --variants attention gated
    python src/ablation.py --skip_baselines          # 只跑 4 种融合

产物
----
    outputs/ablation/results.csv        原始数值（utf-8-sig，Excel 直接打开不乱码）
    outputs/ablation/results.md         Markdown 表格，可直接贴进报告
    outputs/ablation/<variant>/...      每个变体自己的 log/metrics/预测
"""
from __future__ import annotations

import argparse
import csv
import sys

from train_image_only import DEFAULTS as IMAGE_DEFAULTS, run as run_image_only
from train_multimodal import DEFAULTS as MULTIMODAL_DEFAULTS, run as run_multimodal
from train_text_only import DEFAULTS as TEXT_DEFAULTS, run as run_text_only
from utils import Logger, ensure_dir, setup_console

ALL_VARIANTS = ["image_only", "text_only", "early", "late", "attention", "gated"]
FUSION_VARIANTS = {"early", "late", "attention", "gated"}

COLUMNS = [
    ("run_name", "变体"),
    ("modality", "模态"),
    ("fusion", "融合方式"),
    ("test_acc", "Test Acc"),
    ("test_top5", "Test Top-5"),
    ("test_macro_f1", "Macro F1"),
    ("test_macro_precision", "Macro P"),
    ("test_macro_recall", "Macro R"),
    ("val_acc", "Val Acc"),
    ("best_epoch", "Best Epoch"),
    ("train_time_sec", "耗时(s)"),
    ("total_params_M", "参数量(M)"),
    ("trainable_params_M", "可训练(M)"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all ablation variants")
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS)
    parser.add_argument("--epochs", type=int, default=None, help="所有变体共用的 epoch 数")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", type=int, default=None, choices=[0, 1])
    parser.add_argument("--freeze_text", type=int, default=None, choices=[0, 1],
                        help="多模态变体是否冻结 BERT（默认 1）")
    parser.add_argument("--out_dir", default="outputs/ablation")
    parser.add_argument("--skip_baselines", action="store_true")
    return parser


def make_cfg(variant: str, args, out_root: str) -> dict:
    """按变体选择默认配置，再叠加命令行参数。"""
    if variant == "image_only":
        cfg = dict(IMAGE_DEFAULTS)
    elif variant == "text_only":
        cfg = dict(TEXT_DEFAULTS)
    else:
        cfg = dict(MULTIMODAL_DEFAULTS)
        cfg["fusion"] = variant

    cfg["out_dir"] = f"{out_root}/{variant}"
    for key in ("epochs", "batch_size", "seed", "num_workers", "device", "amp"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if variant in FUSION_VARIANTS and args.freeze_text is not None:
        cfg["freeze_text"] = bool(args.freeze_text)
    if variant in FUSION_VARIANTS:
        # 同一组超参，保证消融只差"融合方式"这一个变量
        cfg["epochs"] = int(cfg.get("epochs") or MULTIMODAL_DEFAULTS["epochs"])
    return cfg


def write_table(results: list, out_root: str, logger: Logger) -> None:
    csv_path = ensure_dir(out_root) / "results.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label for _, label in COLUMNS])
        for row in results:
            values = []
            for key, _ in COLUMNS:
                value = row.get(key, "-")
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(value if value is not None else "-")
            writer.writerow(values)
    logger.info(f"[write] {csv_path}")

    baseline = None
    for row in results:
        if row["run_name"] == "image_only":
            baseline = row["test_acc"]
    for row in results:
        if row["run_name"] == "text_only":
            baseline = max(baseline or 0.0, row["test_acc"])
    best_single = baseline

    lines = ["| 变体 | 模态 | 融合 | Test Acc | Top-5 | Macro F1 | Val Acc | 耗时(s) | 可训练参数(M) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in results:
        gap = ""
        if best_single is not None and row["modality"] == "multimodal":
            gap = f" ({row['test_acc'] - best_single:+.4f})"
        lines.append(
            f"| {row['run_name']} | {row['modality']} | {row['fusion']} | "
            f"{row['test_acc']:.4f}{gap} | {fmt(row['test_top5'])} | "
            f"{row['test_macro_f1']:.4f} | {row['val_acc']:.4f} | "
            f"{row['train_time_sec']:.0f} | {row['trainable_params_M']:.2f} |"
        )
    if best_single is not None:
        lines.append("")
        lines.append(f"> 括号内数值 = 相对最佳单模态基线（{best_single:.4f}）的绝对提升。")
    table = "\n".join(lines)
    md_path = ensure_dir(out_root) / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    logger.info(f"[write] {md_path}")
    logger.info("")
    logger.info(table)


def fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main(argv=None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)
    out_root = args.out_dir
    logger = Logger(ensure_dir(out_root) / "ablation.log")
    logger.info("=" * 74)
    logger.info(f"Ablation over: {args.variants}")
    logger.info(f"outputs -> {out_root}")
    logger.info("=" * 74)

    results = []
    for variant in args.variants:
        if args.skip_baselines and variant in ("image_only", "text_only"):
            logger.info(f"[skip] {variant}")
            continue
        cfg = make_cfg(variant, args, out_root)
        logger.info("")
        logger.info(f"########## variant: {variant}  (epochs={cfg['epochs']}) ##########")
        if variant == "image_only":
            result = run_image_only(cfg)
        elif variant == "text_only":
            result = run_text_only(cfg)
        else:
            result = run_multimodal(cfg)
        results.append(result)

        # 释放显存，避免连跑 6 组时 OOM
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if results:
        write_table(results, out_root, logger)
    logger.info("[done] ablation finished.")
    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

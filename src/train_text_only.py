"""train_text_only.py —— 单模态基线 ②：Text-Only（BERT）

对应实验手册 §2.2 与参考脚本 `原/02_singleModal_TextOnly.py`。
与原脚本相比，这里的改进：
  1. 数据来自 annotations.json 的固定 6:2:2 划分（不是每次重新随机划分）；
  2. caption 一次性 tokenize 并缓存，速度快很多；
  3. 加了 label smoothing + cosine 学习率 + 混合精度；
  4. 补上了"验证集选模型"这一步（原脚本只有 train/test，没有 early stop 依据）；
  5. 主干和分类头用不同学习率（BERT 用 2e-5，新初始化的头用 1e-3）。

    python src/train_text_only.py
    python src/train_text_only.py --epochs 6 --batch_size 32
    python src/train_text_only.py --config configs/text_only.yaml

注意：BERT 预训练权重需要联网下载（约 440 MB）。国内可用镜像：
    $env:HF_ENDPOINT = "https://hf-mirror.com"
"""
from __future__ import annotations

import argparse
import sys

from data import build_dataloaders
from model import build_model, describe_model
from text_encoder import load_tokenizer
from trainer import fit
from utils import (Logger, build_config, describe_device, ensure_dir, get_device,
                   set_seed, setup_console)

DEFAULTS = {
    "modality": "text",
    "data_root": "data/cub200_subset",
    "ann_file": "data/cub200_subset/annotations.json",
    "out_dir": "outputs/text_only",
    "text_model": "bert-base-uncased",
    "text_pretrained": True,
    "text_pooling": "mean",
    "freeze_text": False,
    "max_length": 48,
    "epochs": 12,
    "batch_size": 32,
    "lr": 1e-3,
    "backbone_lr": 2e-5,
    "weight_decay": 1e-2,
    "dropout": 0.2,
    "label_smoothing": 0.1,
    "scheduler": "cosine",
    "optimizer": "adamw",
    "amp": True,
    "grad_clip": 1.0,
    "log_every": 20,
    "num_workers": 0,
    "pin_memory": False,
    "seed": 42,
    "device": "cuda",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text-Only baseline (BERT)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--ann_file", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--text_model", default=None)
    parser.add_argument("--text_pooling", default=None, choices=["mean", "cls"])
    parser.add_argument("--freeze_text", type=int, default=None, choices=[0, 1])
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--amp", type=int, default=None, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def run(cfg: dict) -> dict:
    setup_console()
    set_seed(int(cfg["seed"]))
    device = get_device(str(cfg.get("device", "cuda")))
    cfg["device"] = str(device)
    cfg["amp"] = bool(cfg.get("amp", True))
    if cfg.get("freeze_text") is not None:
        cfg["freeze_text"] = bool(cfg["freeze_text"])
    if cfg.get("text_pretrained") is not None:
        cfg["text_pretrained"] = bool(cfg["text_pretrained"])

    out_dir = ensure_dir(cfg["out_dir"])
    logger = Logger(out_dir / "train.log")
    logger.info("=" * 74)
    logger.info("Text-Only baseline (BERT)")
    logger.info(f"device : {describe_device(device)}")
    logger.info(f"model  : {cfg['text_model']}  pooling={cfg['text_pooling']}  "
                f"freeze={cfg.get('freeze_text')}  max_length={cfg['max_length']}")
    logger.info("=" * 74)

    tokenizer = load_tokenizer(cfg["text_model"])
    logger.info(f"[tokenizer] vocab size = {tokenizer.vocab_size}")
    loaders, ann, num_classes = build_dataloaders(cfg, tokenizer=tokenizer, modality="text",
                                                  logger=logger)
    classes = ann["classes"]

    model = build_model(cfg, num_classes).to(device)
    describe_model(model, logger)

    result = fit(model, loaders, num_classes, classes, cfg, logger,
                 out_dir=out_dir, run_name="text_only")
    logger.close()
    return result


def main(argv=None) -> int:
    args = vars(build_parser().parse_args(argv))
    config_path = args.pop("config", None)
    cfg = build_config(DEFAULTS, config_path, args)
    result = run(cfg)
    print("\nSummary:")
    print(f"  text_only   test_acc={result['test_acc']:.4f}  "
          f"macro_f1={result['test_macro_f1']:.4f}  top5={result['test_top5']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

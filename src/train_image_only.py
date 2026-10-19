"""train_image_only.py —— 单模态基线 ①：Image-Only（ResNet18）

对应实验手册 §2.1：输入 = 图像 + 类别标签，模型 = ResNet18，输出 = 类别。
这一步的作用是给后面的多模态实验提供"只有图像能到多少"的下界/参照。

    python src/train_image_only.py
    python src/train_image_only.py --epochs 5 --batch_size 64
    python src/train_image_only.py --config configs/image_only.yaml

产物（默认 outputs/image_only/）
    train.log              完整训练日志（可直接截取到报告）
    checkpoints/best.pt    验证集最优权重
    metrics.json           测试集全部指标（含 100 类逐类准确率）
    history.json           每个 epoch 的 loss/acc，用来画训练曲线
    test_predictions.npz   测试集预测结果（错误分析用）
"""
from __future__ import annotations

import argparse
import sys

from data import build_dataloaders
from model import build_model, describe_model
from trainer import fit
from utils import (Logger, build_config, ensure_dir, get_device, describe_device,
                   set_seed, setup_console)

DEFAULTS = {
    "modality": "image",
    "data_root": "data/cub200_subset",
    "ann_file": "data/cub200_subset/annotations.json",
    "out_dir": "outputs/image_only",
    "image_backbone": "resnet18",
    "image_pretrained": True,
    "image_size": 224,
    "epochs": 15,
    "batch_size": 32,
    "lr": 3e-4,
    "backbone_lr": None,
    "weight_decay": 1e-4,
    "dropout": 0.2,
    "label_smoothing": 0.1,
    "scheduler": "cosine",
    "optimizer": "adamw",
    "amp": True,
    "grad_clip": 1.0,
    "log_every": 20,
    "num_workers": 4,
    "pin_memory": True,
    "seed": 42,
    "device": "cuda",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image-Only baseline (ResNet18)")
    parser.add_argument("--config", default=None, help="yaml 配置文件")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--ann_file", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--image_backbone", default=None, choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--image_pretrained", type=int, default=None, choices=[0, 1])
    parser.add_argument("--image_size", type=int, default=None)
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
    if cfg.get("image_pretrained") is not None:
        cfg["image_pretrained"] = bool(cfg["image_pretrained"])
    cfg["amp"] = bool(cfg.get("amp", True))

    out_dir = ensure_dir(cfg["out_dir"])
    logger = Logger(out_dir / "train.log")
    logger.info("=" * 74)
    logger.info("Image-Only baseline (ResNet18)")
    logger.info(f"device : {describe_device(device)}")
    logger.info(f"data   : {cfg['data_root']}")
    logger.info("=" * 74)

    loaders, ann, num_classes = build_dataloaders(cfg, tokenizer=None, modality="image", logger=logger)
    classes = ann["classes"]

    model = build_model(cfg, num_classes).to(device)
    describe_model(model, logger)

    result = fit(model, loaders, num_classes, classes, cfg, logger,
                 out_dir=out_dir, run_name="image_only")
    logger.close()
    return result


def main(argv=None) -> int:
    args = vars(build_parser().parse_args(argv))
    config_path = args.pop("config", None)
    cfg = build_config(DEFAULTS, config_path, args)
    result = run(cfg)
    print("\nSummary:")
    print(f"  image_only  test_acc={result['test_acc']:.4f}  "
          f"macro_f1={result['test_macro_f1']:.4f}  top5={result['test_top5']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

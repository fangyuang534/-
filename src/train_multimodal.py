"""train_multimodal.py —— 多模态训练：图像 + 文本 -> 融合 -> 分类

换融合方式只改一个字段：
    python src/train_multimodal.py --fusion early
    python src/train_multimodal.py --fusion late
    python src/train_multimodal.py --fusion attention     # 推荐
    python src/train_multimodal.py --fusion gated
或者直接用配置文件（等价）：
    python src/train_multimodal.py --config configs/fusion_attention.yaml

默认设置（8 GB 显存的笔记本也能跑）：
    * 图像主干 ResNet18：微调（freeze_image=0）
    * 文本主干 BERT：**冻结**（freeze_text=1），只训练投影 + 融合 + 分类头
      -> 这样每个 epoch 约 1 分钟，四种融合方式半小时内全部跑完
      -> 想追更高指标可以 --freeze_text 0（用 backbone_lr 2e-5 微调 BERT，速度慢约 3 倍）
    * 两个模态都先投影到 proj_dim=256 的公共维度，再做融合

产物（默认 outputs/fusion_<name>/）
    train.log / checkpoints/best.pt / metrics.json / history.json
    test_predictions.npz   （含 gate：门控/晚融合的权重，用来看"更信谁"）
"""
from __future__ import annotations

import argparse
import sys

from data import build_dataloaders
from fusion import FUSION_DESCRIPTIONS
from model import build_model, describe_model
from text_encoder import load_tokenizer
from trainer import fit
from utils import (Logger, build_config, describe_device, ensure_dir, get_device,
                   set_seed, setup_console)

DEFAULTS = {
    "modality": "multimodal",
    "data_root": "data/cub200_subset",
    "ann_file": "data/cub200_subset/annotations.json",
    "out_dir": "outputs/fusion_attention",
    "fusion": "attention",
    "proj_dim": 256,
    "num_heads": 8,
    "image_backbone": "resnet18",
    "image_pretrained": True,
    "freeze_image": False,
    "text_model": "bert-base-uncased",
    "text_pretrained": True,
    "freeze_text": True,
    "text_pooling": "mean",
    "max_length": 48,
    "image_size": 224,
    "epochs": 8,
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
    parser = argparse.ArgumentParser(description="Multimodal bird classification (4 fusion types)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--ann_file", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--fusion", default=None,
                        choices=["early", "late", "attention", "gated",
                                 "concat", "cross_attention", "gate"])
    parser.add_argument("--proj_dim", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--image_backbone", default=None, choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--image_pretrained", type=int, default=None, choices=[0, 1])
    parser.add_argument("--freeze_image", type=int, default=None, choices=[0, 1])
    parser.add_argument("--text_model", default=None)
    parser.add_argument("--freeze_text", type=int, default=None, choices=[0, 1])
    parser.add_argument("--text_pooling", default=None, choices=["mean", "cls"])
    parser.add_argument("--max_length", type=int, default=None)
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
    for key in ("image_pretrained", "text_pretrained", "freeze_image", "freeze_text", "amp"):
        if cfg.get(key) is not None:
            cfg[key] = bool(cfg[key])

    # 变体名（别名归一化），用于 out_dir 与日志
    fusion_key = str(cfg["fusion"]).lower()
    alias = {"concat": "early", "cross_attention": "attention", "gate": "gated"}
    fusion_key = alias.get(fusion_key, fusion_key)
    cfg["fusion"] = fusion_key
    if not cfg.get("out_dir") or cfg["out_dir"].endswith("fusion_attention"):
        cfg["out_dir"] = f"outputs/fusion_{fusion_key}"

    out_dir = ensure_dir(cfg["out_dir"])
    logger = Logger(out_dir / "train.log")
    logger.info("=" * 74)
    logger.info(f"Multimodal fusion = {fusion_key}")
    logger.info(f"  {FUSION_DESCRIPTIONS.get(fusion_key, '')}")
    logger.info(f"device : {describe_device(device)}")
    logger.info(f"freeze : image={cfg.get('freeze_image')}  text={cfg.get('freeze_text')}  "
                f"proj_dim={cfg.get('proj_dim')}")
    logger.info("=" * 74)

    tokenizer = load_tokenizer(cfg["text_model"])
    loaders, ann, num_classes = build_dataloaders(cfg, tokenizer=tokenizer,
                                                  modality="multimodal", logger=logger)
    classes = ann["classes"]

    model = build_model(cfg, num_classes).to(device)
    describe_model(model, logger)

    result = fit(model, loaders, num_classes, classes, cfg, logger,
                 out_dir=out_dir, run_name=f"fusion_{fusion_key}", collect=("gate",))
    logger.close()
    return result


def main(argv=None) -> int:
    args = vars(build_parser().parse_args(argv))
    config_path = args.pop("config", None)
    cfg = build_config(DEFAULTS, config_path, args)
    result = run(cfg)
    print("\nSummary:")
    print(f"  fusion={result['fusion']:10s} test_acc={result['test_acc']:.4f}  "
          f"macro_f1={result['test_macro_f1']:.4f}  top5={result['test_top5']:.4f}  "
          f"time={result['train_time_sec']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

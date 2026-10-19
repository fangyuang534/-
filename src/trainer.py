"""trainer.py —— 通用训练流程（三个训练脚本共用，避免重复代码）。

整体流程
--------
    for epoch in 1..E:
        train_one_epoch            # 训练一个 epoch，打印 loss / train_acc
        evaluate(val)              # 验证集算 accuracy / macro-F1
        若 val_accuracy 创新高  -> 保存 checkpoints/best.pt
    载入 best.pt -> 在 test 上评估 -> 保存 metrics.json / test_predictions.npz / history.json
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

from engine import predict, train_one_epoch
from metrics import compute_metrics, print_metrics
from utils import count_parameters, format_seconds, save_json


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """AdamW；可选 backbone_lr（主干用小学习率，新初始化的头用大学习率）。"""
    lr = float(cfg.get("lr", 1e-4))
    backbone_lr = cfg.get("backbone_lr")
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    name = cfg.get("optimizer", "adamw").lower()

    param_groups = None
    if backbone_lr:
        backbone_params, head_params = [], []
        for pname, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if pname.startswith("encoder") or pname.startswith("image_encoder") \
                    or pname.startswith("text_encoder"):
                backbone_params.append(param)
            else:
                head_params.append(param)
        param_groups = [
            {"params": backbone_params, "lr": float(backbone_lr)},
            {"params": head_params, "lr": lr},
        ]

    if name == "sgd":
        if param_groups is None:
            param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if param_groups is None:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, cfg: dict, epochs: int):
    name = cfg.get("scheduler", "cosine").lower()
    if name == "none" or epochs <= 1:
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def fit(model: nn.Module, loaders: dict, num_classes: int, classes: list, cfg: dict,
        logger, out_dir, run_name: str = "run", collect: tuple = ("gate",)) -> dict:
    """完整的"训练 -> 选最优 -> 测试"流程，返回可直接写进消融表的指标 dict。"""
    device = torch.device(cfg["device"])
    epochs = int(cfg["epochs"])
    amp = bool(cfg.get("amp", True))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    log_every = int(cfg.get("log_every", 20))

    out_dir = str(out_dir)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.1)))
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, epochs)

    total_params, trainable_params = count_parameters(model)
    logger.info(f"[model] total params={total_params / 1e6:.2f} M, "
                f"trainable={trainable_params / 1e6:.2f} M")
    logger.info(f"[model] optimizer={cfg.get('optimizer', 'adamw')}  lr={cfg.get('lr')}  "
                f"backbone_lr={cfg.get('backbone_lr')}  epochs={epochs}  amp={amp}")

    best_acc, best_epoch, best_path = -1.0, 0, os.path.join(ckpt_dir, "best.pt")
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        logger.info(f"===== epoch {epoch}/{epochs} =====")
        train_stats = train_one_epoch(model, loaders["train"], criterion, optimizer, device,
                                      epoch, epochs, logger, amp=amp,
                                      grad_clip=grad_clip, log_every=log_every)

        val_raw = predict(model, loaders["val"], device, amp=amp)
        val_metrics = compute_metrics(val_raw["y_true"], val_raw["y_pred"], val_raw["probs"],
                                     num_classes=num_classes, classes=classes)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[epoch {epoch}] train_loss={train_stats['loss']:.4f}  "
                    f"train_acc={train_stats['acc']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
                    f"val_f1={val_metrics['macro_f1']:.4f}  "
                    f"val_top5={val_metrics['top5_accuracy']:.4f}  "
                    f"lr={current_lr:.2e}  time={format_seconds(train_stats['seconds'])}")

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_top5_acc": val_metrics["top5_accuracy"],
            "lr": current_lr,
            "seconds": train_stats["seconds"],
        })

        if val_metrics["accuracy"] > best_acc:
            best_acc, best_epoch = val_metrics["accuracy"], epoch
            torch.save({
                "model": model.state_dict(),
                "cfg": {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool, type(None)))},
                "num_classes": num_classes,
                "classes": classes,
                "epoch": epoch,
                "val_accuracy": best_acc,
            }, best_path)
            logger.info(f"[save] new best val_acc={best_acc:.4f} -> {best_path}")
        if scheduler is not None:
            scheduler.step()

    timer_elapsed = time.time() - start_time
    elapsed = timer_elapsed

    # ---------------- 用最优权重在 test 上评估 ----------------
    logger.info(f"[test] loading best checkpoint (epoch {best_epoch}, val_acc={best_acc:.4f})")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    test_raw = predict(model, loaders["test"], device, amp=amp, collect=collect)
    test_metrics = compute_metrics(test_raw["y_true"], test_raw["y_pred"], test_raw["probs"],
                                  num_classes=num_classes, classes=classes)
    print_metrics(test_metrics, title=f"TEST / {run_name}", logger=logger)

    # ---------------- 保存产物 ----------------
    save_json(test_metrics, os.path.join(out_dir, "metrics.json"))
    save_json(history, os.path.join(out_dir, "history.json"))
    save_json({k: v for k, v in cfg.items()}, os.path.join(out_dir, "config.json"))

    npz_payload = {
        "y_true": test_raw["y_true"],
        "y_pred": test_raw["y_pred"],
        "probs": test_raw["probs"].astype(np.float32),
        "index": test_raw["index"],
        "paths": test_raw["paths"],
        "captions": test_raw["captions"],
    }
    if "gate" in test_raw:
        npz_payload["gate"] = np.asarray(test_raw["gate"]).reshape(len(test_raw["y_pred"]), -1)
    np.savez_compressed(os.path.join(out_dir, "test_predictions.npz"), **npz_payload)

    result = {
        "run_name": run_name,
        "modality": cfg.get("modality"),
        "fusion": cfg.get("fusion", "-"),
        "test_acc": test_metrics["accuracy"],
        "test_top5": test_metrics["top5_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "val_acc": best_acc,
        "best_epoch": best_epoch,
        "train_time_sec": round(elapsed, 1),
        "total_params_M": round(total_params / 1e6, 2),
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "out_dir": out_dir,
    }
    save_json(result, os.path.join(out_dir, "result.json"))
    logger.info(f"[done] {run_name}: test_acc={result['test_acc']:.4f}  "
                f"test_macro_f1={result['test_macro_f1']:.4f}  "
                f"time={format_seconds(elapsed)}")
    logger.info(f"[done] outputs -> {out_dir}")
    return result

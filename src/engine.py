"""engine.py —— 训练/评估循环（三个训练脚本共用）。

设计要点
--------
1. forward_batch：按 batch 里**有哪些 key** 决定传什么给模型，
   所以同一个循环能同时支持 Image-Only / Text-Only / 多模态。
2. 混合精度（AMP）：GPU 上自动用 fp16 + GradScaler，速度约快 1.5~2 倍，
   显存也省一半（8 GB 的 4060 也能跑 batch_size=32）。
3. 评估时把 probs / image_path / 融合模块的附加输出（attention 权重、gate）
   一起收集下来，存成 npz，后面 error_analysis.py 和报告可视化直接读。
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from metrics import compute_metrics
from utils import AverageMeter, make_grad_scaler


def forward_batch(model: nn.Module, batch: dict, device: torch.device):
    kwargs = {}
    if "images" in batch:
        kwargs["images"] = batch["images"].to(device, non_blocking=True)
    if "input_ids" in batch:
        kwargs["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
        kwargs["attention_mask"] = batch["attention_mask"].to(device, non_blocking=True)
    return model(**kwargs)


def get_logits(output) -> torch.Tensor:
    if isinstance(output, dict):
        return output["logits"]
    return output


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: torch.device,
                    epoch: int, epochs: int, logger, amp: bool = True,
                    grad_clip: float = 1.0, log_every: int = 20) -> dict:
    """跑一个 epoch，返回 {"loss":..., "acc":..., "seconds":...}"""
    model.train()
    scaler = make_grad_scaler(device, enabled=amp)
    loss_meter, acc_meter = AverageMeter("loss"), AverageMeter("acc")
    use_amp = bool(amp and device.type == "cuda")
    start = time.time()

    for step, batch in enumerate(loader, 1):
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = forward_batch(model, batch, device)
            logits = get_logits(output)
            loss = criterion(logits, labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            preds = logits.detach().argmax(dim=1)
            acc = (preds == labels).float().mean().item()
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(acc, labels.size(0))

        if log_every and step % log_every == 0:
            logger.info(f"  epoch {epoch}/{epochs}  step {step}/{len(loader)}  "
                        f"loss={loss_meter.avg:.4f}  acc={acc_meter.avg:.4f}")

    return {"loss": loss_meter.avg, "acc": acc_meter.avg, "seconds": time.time() - start}


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device, amp: bool = True,
            collect: tuple = ()) -> dict:
    """在给定 dataloader 上做推理，返回所有预测结果。

    collect: 需要额外收集的字段名（例如 ("attn_weights", "gate")），
             只有多模态模型的输出 dict 里才可能有。
    """
    model.eval()
    use_amp = bool(amp and device.type == "cuda")
    all_logits, all_labels, all_index = [], [], []
    all_paths, all_captions = [], []
    collected = {key: [] for key in collect}

    for batch in loader:
        labels = batch["label"]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = forward_batch(model, batch, device)
        logits = get_logits(output).detach().float().cpu()
        all_logits.append(logits)
        all_labels.append(labels.clone())
        all_index.append(batch["index"].clone())
        if "image_path" in batch:
            all_paths.extend(list(batch["image_path"]))
        if "caption" in batch:
            all_captions.extend(list(batch["caption"]))
        if isinstance(output, dict):
            for key in collect:
                value = output.get(key)
                if value is not None:
                    collected[key].append(value.detach().float().cpu())

    logits = torch.cat(all_logits, dim=0)                     # [N, C]
    labels = torch.cat(all_labels, dim=0)
    probs = torch.softmax(logits, dim=1)
    result = {
        "logits": logits.numpy(),
        "probs": probs.numpy(),
        "y_true": labels.numpy().astype(np.int64),
        "y_pred": logits.argmax(dim=1).numpy().astype(np.int64),
        "index": torch.cat(all_index, dim=0).numpy().astype(np.int64),
        "paths": np.array(all_paths) if all_paths else np.array([]),
        "captions": np.array(all_captions) if all_captions else np.array([]),
    }
    for key, chunks in collected.items():
        try:
            result[key] = torch.cat(chunks, dim=0).numpy()
        except RuntimeError:
            # 形状不一致（例如文本长度可变）时按对象数组保存
            result[key] = np.array([c.numpy() for c in chunks], dtype=object)
    return result


def evaluate(model: nn.Module, loader, device: torch.device, num_classes: int,
             classes: list = None, amp: bool = True, collect: tuple = ()) -> tuple:
    """推理 + 计算指标，返回 (metrics, raw_predictions)。"""
    raw = predict(model, loader, device, amp=amp, collect=collect)
    metrics = compute_metrics(raw["y_true"], raw["y_pred"], raw["probs"],
                              num_classes=num_classes, classes=classes)
    return metrics, raw

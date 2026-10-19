"""data.py —— 数据集与 DataLoader。

一个样本 = {图片, 该图的 10 条描述之一, 类别标签}
------------------------------------------------
* 同一个 __getitem__ 可以按需返回"图像 / 文本 / 图像+文本"，
  三个训练脚本共用同一套 DataLoader，只是 modality 不同。
* 训练时**随机挑一条** caption（数据增强：同一张图的不同描述），
  评估时固定用第一条，保证多次评估结果可复现。
* caption 在 __init__ 里一次性 tokenize 并缓存（pretokenize=True），
  否则每个 epoch 都要重复分词，速度会慢很多。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import project_path

IMAGE_MEAN = (0.485, 0.456, 0.406)   # ImageNet 统计量（ResNet/BERT 都按这个做图像标准化）
IMAGE_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# 图像变换
# ----------------------------------------------------------------------------
def build_transform(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(round(image_size * 256 / 224))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])


# ----------------------------------------------------------------------------
# 标注读取
# ----------------------------------------------------------------------------
def load_annotations(ann_file) -> dict:
    path = project_path(ann_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到标注文件：{path}\n"
            f"请先运行：python src/build_annotations.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_num_classes(ann: dict) -> int:
    return int(ann.get("meta", {}).get("num_classes") or len(ann["classes"]))


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class BirdDataset(Dataset):
    """多模态鸟类数据集。

    Args:
        data_root:  解压后的数据根目录（images/ 与 text/ 的父目录）
        samples:    annotations.json 里 splits[split] 的列表
        tokenizer:  HuggingFace tokenizer；为 None 时只返回图像（Image-Only 用）
        max_length: 文本最大 token 数（CUB 的 caption 约 10~25 个词，48 足够）
        train:      训练模式：随机裁剪/翻转 + 随机选一条 caption
        image_size: 输入分辨率
        load_image: 是否真的读图（Text-Only 可以直接设 False，省掉 IO）
    """

    def __init__(self, data_root, samples: list, tokenizer=None, max_length: int = 48,
                 train: bool = True, image_size: int = 224, load_image: bool = True,
                 pretokenize: bool = True, indices: list = None):
        self.data_root = Path(data_root)
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = train
        self.load_image = load_image
        self.transform = build_transform(image_size, train=train)
        self.indices = list(range(len(self.samples))) if indices is None else list(indices)
        # indices 用来把"第 i 个样本"映射回 splits 里的原始位置（错误分析时对齐用）

        self.tokenized = None
        if tokenizer is not None and pretokenize:
            self.tokenized = [self._tokenize_all(rec["captions"]) for rec in self.samples]

    # ---- 内部工具 ----
    def _tokenize_all(self, captions: list) -> list:
        """把一条样本的全部 caption 一起编码（一次 forward 处理 10 条，比逐条快）。"""
        enc = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}
                for i in range(len(captions))]

    def _pick_caption_index(self, n: int) -> int:
        if n == 1 or not self.train:
            return 0
        return int(torch.randint(0, n, (1,)).item())

    # ---- Dataset 接口 ----
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        rec = self.samples[index]
        item = {
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "index": torch.tensor(self.indices[index], dtype=torch.long),
            "image_path": rec["image_path"],
        }

        if self.load_image:
            image_path = self.data_root / rec["image_path"]
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            item["images"] = self.transform(image)

        if self.tokenizer is not None:
            n = len(rec["captions"])
            cap_idx = self._pick_caption_index(n)
            tokens = (self.tokenized[index][cap_idx] if self.tokenized is not None
                      else self._tokenize_all(rec["captions"])[cap_idx])
            item["input_ids"] = tokens["input_ids"]
            item["attention_mask"] = tokens["attention_mask"]
            item["caption"] = rec["captions"][cap_idx]
        return item


# ----------------------------------------------------------------------------
# DataLoader 工厂
# ----------------------------------------------------------------------------
def build_dataloaders(cfg: dict, tokenizer=None, modality: str = "multimodal",
                      logger=None) -> tuple:
    """按 modality 组装 train/val/test 三个 DataLoader。

    modality:
        "image"      -> 只返回图片（Image-Only）
        "text"       -> 只返回文本（Text-Only），不读图
        "multimodal" -> 图片 + 文本
    """
    ann = load_annotations(cfg["ann_file"])
    num_classes = get_num_classes(ann)
    data_root = cfg["data_root"]
    image_size = int(cfg.get("image_size", 224))
    max_length = int(cfg.get("max_length", 48))
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 4))
    load_image = modality in ("image", "multimodal")
    use_text = modality in ("text", "multimodal")

    datasets = {}
    for split in ("train", "val", "test"):
        samples = ann["splits"][split]
        # 给每个样本一个全局唯一的 index（train 在前，test 在后），错误分析要对齐
        offset = {"train": 0, "val": len(ann["splits"]["train"]),
                  "test": len(ann["splits"]["train"]) + len(ann["splits"]["val"])}[split]
        datasets[split] = BirdDataset(
            data_root=data_root,
            samples=samples,
            tokenizer=tokenizer if use_text else None,
            max_length=max_length,
            train=(split == "train"),
            image_size=image_size,
            load_image=load_image,
            indices=list(range(offset, offset + len(samples))),
        )

    pin_memory = bool(cfg.get("pin_memory", True)) and torch.cuda.is_available()
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory),
    }
    if logger is not None:
        logger.info(f"[data] modality={modality}  classes={num_classes}  "
                    f"train/val/test={len(datasets['train'])}/{len(datasets['val'])}/"
                    f"{len(datasets['test'])}  batch_size={batch_size}")
    return loaders, ann, num_classes

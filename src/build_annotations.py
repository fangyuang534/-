"""build_annotations.py —— 数据准备：解压数据集 → 扫描图文对 → 6:2:2 划分 → annotations.json

数据集原始结构（CUB_200_subset.zip 里的样子）
---------------------------------------------
    CUB_200_subset/
    ├── images/
    │   ├── 001.Black_footed_Albatross/
    │   │   ├── Black_Footed_Albatross_0001_796111.jpg
    │   │   └── ...
    │   └── ...（共 100 个类别文件夹）
    └── text/                                  ← 注意：描述文本不在 images 旁边，
        ├── 001.Black_footed_Albatross/          而是在和 images 平级的 text/ 里，
        │   ├── Black_Footed_Albatross_0001_796111.txt   **文件名（不含后缀）与图片同名**
        │   └── ...
        └── ...

每个 txt 有 10 行，一行一条英文描述（caption），例如：
    the medium sized bird has a dark grey color, a black downward curved beak, and long wings.

生成的标注文件（本脚本产出）
--------------------------
    data/cub200_subset/
    ├── annotations.json            ← 主标注：meta / classes / class_prompts / splits
    ├── annotations_train.json      ← 兼容原参考脚本 02_singleModal_TextOnly.py 的格式
    ├── annotations_val.json
    ├── annotations_test.json
    ├── images/...
    └── text/...

annotations.json 结构
--------------------
{
  "meta": {"source": ..., "num_classes": 100, "num_captions": 10,
           "num_images": 5864, "split_ratio": [0.6, 0.2, 0.2], "seed": 42,
           "split_sizes": {"train": 3518, "val": 1173, "test": 1173}},
  "classes": ["001.Black_footed_Albatross", ...],          # 下标 == label
  "class_prompts": {"001.Black_footed_Albatross": "a photo of a black footed albatross"},
  "splits": {"train": [{"image_path": "images/001.../xxx.jpg",
                        "captions": ["...", ...10 条],
                        "label": 0}, ...],
             "val": [...], "test": [...]}
}

用法
----
    python src/build_annotations.py                 # 自动在 数据集/ 等位置查找 zip
    python src/build_annotations.py --zip D:/data/CUB_200_subset.zip
    python src/build_annotations.py --skip_extract  # 已经解压过，只重建标注

数据不随仓库分发：把 zip 放到 `数据集/` 下，或用环境变量 CUB200_ZIP 指定即可，
详见 docs/DATASET.md。本脚本只依赖标准库（不需要 torch），在装依赖之前就能先跑。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from paths import (Logger, find_dataset_zip, project_path, setup_console,
                   zip_search_report)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_PREFIX = re.compile(r"^\d+[\.\-_]\s*")     # "001.Black_footed_Albatross" -> "Black_footed_Albatross"


# ----------------------------------------------------------------------------
# 1. 解压
# ----------------------------------------------------------------------------
def extract_zip(zip_path: Path, out_dir: Path, logger: Logger) -> None:
    """把 zip 解压到 out_dir（带进度条）。已解压则跳过。"""
    marker = out_dir / "images"
    if marker.is_dir() and any(marker.iterdir()):
        logger.info(f"[skip] already extracted: {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        total = len(members)
        logger.info(f"[extract] {zip_path.name} -> {out_dir}  ({total} entries)")
        for i, member in enumerate(members, 1):
            zf.extract(member, out_dir)
            if i % 500 == 0 or i == total:
                logger.info(f"          {i}/{total} ({i / total * 100:5.1f}%)")
    logger.info("[extract] done.")


# ----------------------------------------------------------------------------
# 2. 扫描图文对
# ----------------------------------------------------------------------------
def class_display_name(folder_name: str) -> str:
    """"001.Black_footed_Albatross" -> "black footed albatross"（给 CLIP 提示词用）"""
    name = CLASS_PREFIX.sub("", folder_name)
    name = name.replace("_", " ").replace("-", " ").strip()
    return re.sub(r"\s+", " ", name).lower()


def read_captions(txt_path: Path, max_captions: int) -> list:
    """读一个 txt，返回非空行列表（最多 max_captions 条）。"""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f.read().splitlines()]
    except OSError:
        return []
    captions = [line for line in lines if line]
    return captions[:max_captions]


def scan_pairs(data_root: Path, max_captions: int, logger: Logger) -> tuple:
    """扫描 images/ 与 text/，返回 (classes, records)；records 里已带 label。"""
    images_root = data_root / "images"
    text_root = data_root / "text"
    if not images_root.is_dir():
        raise FileNotFoundError(f"解压后没有 images/ 目录：{images_root}")

    class_dirs = sorted([d for d in images_root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"{images_root} 下没有任何类别文件夹。")
    classes = [d.name for d in class_dirs]
    class_to_label = {name: idx for idx, name in enumerate(classes)}
    logger.info(f"[scan] {len(classes)} classes")

    records = []
    missing_caption = []
    caption_lens = []
    for class_dir in class_dirs:
        label = class_to_label[class_dir.name]
        text_dir = text_root / class_dir.name
        image_files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        for image_path in image_files:
            txt_path = text_dir / f"{image_path.stem}.txt"
            captions = read_captions(txt_path, max_captions) if txt_path.is_file() else []
            if not captions:
                missing_caption.append(str(image_path.relative_to(data_root)))
                continue
            caption_lens.extend(len(c.split()) for c in captions)
            records.append({
                "image_path": image_path.relative_to(data_root).as_posix(),
                "captions": captions,
                "label": label,
            })

    logger.info(f"[scan] image-text pairs: {len(records)}")
    if missing_caption:
        logger.info(f"[scan] images WITHOUT caption (dropped): {len(missing_caption)}")
        for p in missing_caption[:5]:
            logger.info(f"        e.g. {p}")
    if caption_lens:
        mean_len = sum(caption_lens) / len(caption_lens)
        logger.info(f"[scan] caption words: mean={mean_len:.1f}, "
                    f"min={min(caption_lens)}, max={max(caption_lens)}")
    return classes, records


# ----------------------------------------------------------------------------
# 3. 按类别分层 6:2:2 划分
# ----------------------------------------------------------------------------
def stratified_split(records: list, num_classes: int, ratios: tuple, seed: int, logger: Logger) -> dict:
    """每个类别内部先打乱再按比例切分，保证 train/val/test 类别分布一致。"""
    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["label"]].append(rec)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for label in range(num_classes):
        items = sorted(by_class.get(label, []), key=lambda r: r["image_path"])
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        # 保证每一份至少 1 张（类别样本极少时才可能触发）
        n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, n - 1)
        n_val = max(1, min(n_val, n - n_train - 1)) if n - n_train >= 2 else max(0, n - n_train - 1)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])

    for name in splits:
        rng.shuffle(splits[name])
        counter = Counter(rec["label"] for rec in splits[name])
        logger.info(f"[split] {name:5s}: {len(splits[name]):5d} images, "
                    f"{len(counter)} classes present")
    return splits


# ----------------------------------------------------------------------------
# 4. 写标注文件
# ----------------------------------------------------------------------------
def write_annotations(out_path: Path, classes: list, class_prompts: dict, splits: dict,
                      meta: dict, logger: Logger) -> None:
    payload = {
        "meta": meta,
        "classes": classes,
        "class_prompts": class_prompts,
        "splits": splits,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    size_mb = out_path.stat().st_size / 1024 ** 2
    logger.info(f"[write] {out_path}  ({size_mb:.1f} MB)")


def write_legacy_splits(data_root: Path, splits: dict, logger: Logger) -> None:
    """额外导出 annotations_{train,val,test}.json，格式与参考脚本 02 一致。"""
    for name, items in splits.items():
        payload = {"annotations": [
            {"image_path": rec["image_path"], "captions": rec["captions"], "label": rec["label"]}
            for rec in items
        ]}
        path = data_root / f"annotations_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        logger.info(f"[write] {path.name} ({len(items)} items)")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CUB_200_subset 数据准备与标注生成")
    parser.add_argument("--zip", default=None,
                        help="原始压缩包路径；不填则自动查找 数据集/CUB_200_subset.zip、"
                             "仓库根目录、或环境变量 CUB200_ZIP")
    parser.add_argument("--out", default="data/cub200_subset", help="解压 + 标注输出目录（即 data_root）")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_captions", type=int, default=10, help="每张图最多保留几条描述")
    parser.add_argument("--skip_extract", action="store_true", help="数据已解压时跳过解压")
    parser.add_argument("--no_legacy", action="store_true", help="不导出 annotations_{split}.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    setup_console()
    args = parse_args(argv)

    data_root = project_path(args.out)
    logger = Logger(data_root / "build_annotations.log")
    logger.info("=" * 78)
    logger.info("CUB_200_subset -> annotations.json")
    logger.info("=" * 78)

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        logger.info(f"[error] ratios must sum to 1.0, got {ratios}")
        return 1

    zip_path = None
    if not args.skip_extract:
        zip_path = find_dataset_zip(args.zip)
        if zip_path is None:
            logger.info("[error] dataset zip not found. searched:")
            logger.info(zip_search_report())
            logger.info("hint: the dataset is NOT included in this repository "
                        "(see docs/DATASET.md).")
            logger.close()
            return 1
        logger.info(f"[zip] {zip_path}")
        extract_zip(zip_path, data_root, logger)

    classes, records = scan_pairs(data_root, args.num_captions, logger)
    if not records:
        logger.info("[error] no image-text pair found.")
        return 1

    class_prompts = {name: f"a photo of a {class_display_name(name)}" for name in classes}
    splits = stratified_split(records, len(classes), ratios, args.seed, logger)

    caption_counts = Counter(len(rec["captions"]) for rec in records)
    meta = {
        "source": zip_path.name if zip_path else "(skipped extraction)",
        "num_classes": len(classes),
        "num_images": len(records),
        "num_captions": args.num_captions,
        "caption_count_distribution": {str(k): v for k, v in sorted(caption_counts.items())},
        "split_ratio": list(ratios),
        "seed": args.seed,
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "note": "captions 来自 text/<class>/<stem>.txt；与 images/<class>/<stem>.jpg 同名配对",
    }

    write_annotations(data_root / "annotations.json", classes, class_prompts, splits, meta, logger)
    if not args.no_legacy:
        write_legacy_splits(data_root, splits, logger)

    logger.info("-" * 78)
    logger.info(f"classes : {len(classes)}")
    logger.info(f"images  : {len(records)}")
    for name in ("train", "val", "test"):
        logger.info(f"{name:7s} : {len(splits[name])}")
    logger.info(f"example prompt: {classes[0]!r} -> {class_prompts[classes[0]]!r}")
    logger.info("done. next:  python src/train_image_only.py")
    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

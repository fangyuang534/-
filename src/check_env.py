"""check_env.py —— 30 秒自检：环境装好没、数据准备好了没、GPU 能不能用。

建议每次开始实验前先跑一遍：
    python src/check_env.py

它不会训练任何东西，只做检查并给出"下一步该跑什么"的提示。
注意：本脚本只依赖标准库 + paths.py，所以**在 torch 还没装好时也能跑**。
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

from paths import (PROJECT_ROOT, REPO_ROOT, find_dataset_zip, project_path,
                   setup_console, zip_search_report)

OK = "[ OK ]"
NO = "[MISS]"


def check_python_packages() -> list:
    rows = []
    packages = ["torch", "torchvision", "transformers", "PIL", "numpy",
                "yaml", "tqdm", "matplotlib", "sklearn", "pandas"]
    for name in packages:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            rows.append((name, OK, str(version)))
        except Exception as exc:                     # noqa: BLE001
            required = name in ("torch", "torchvision", "transformers", "PIL", "numpy", "yaml")
            rows.append((name, NO if required else "[opt ]", f"not installed ({type(exc).__name__})"))
    return rows


def check_cuda() -> list:
    rows = []
    try:
        import torch
    except Exception:
        return [("torch", NO, "cannot import torch, install it first")]
    rows.append(("torch version", OK, torch.__version__))
    rows.append(("cuda built", OK if torch.version.cuda else "[cpu ]", str(torch.version.cuda)))
    rows.append(("cuda available", OK if torch.cuda.is_available() else "[cpu ]",
                 str(torch.cuda.is_available())))
    if torch.cuda.is_available():
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        rows.append(("gpu", OK, f"{torch.cuda.get_device_name(idx)} "
                                f"({props.total_memory / 1024 ** 3:.1f} GB)"))
        try:
            x = torch.randn(1024, 1024, device="cuda")
            y = (x @ x).sum().item()
            rows.append(("gpu matmul", OK, f"ok ({y:.1f})"))
        except Exception as exc:                     # noqa: BLE001
            rows.append(("gpu matmul", NO, f"{type(exc).__name__}: {exc}"))
    return rows


def check_data() -> list:
    rows = []
    zip_path = find_dataset_zip()
    if zip_path is not None:
        size_mb = zip_path.stat().st_size / 1024 ** 2
        rows.append(("dataset zip", OK, f"{zip_path}  ({size_mb:.0f} MB)"))
    else:
        rows.append(("dataset zip", "[info ]", "not found -> see docs/DATASET.md"))
        rows.append(("  searched", "[info ]", zip_search_report().replace("\n", "\n            ").strip()))
    data_root = project_path("data/cub200_subset")
    images_dir = data_root / "images"
    if images_dir.is_dir():
        num_classes = len([d for d in images_dir.iterdir() if d.is_dir()])
        rows.append(("data/images", OK, f"{num_classes} class folders under {images_dir}"))
    else:
        rows.append(("data/images", "[info ]", "not extracted yet -> run build_annotations.py"))
    ann_path = data_root / "annotations.json"
    if ann_path.is_file():
        import json
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        sizes = ann.get("meta", {}).get("split_sizes", {})
        rows.append(("annotations.json", OK,
                     f"classes={len(ann.get('classes', []))} "
                     f"train/val/test={sizes.get('train')}/{sizes.get('val')}/{sizes.get('test')}"))
    else:
        rows.append(("annotations.json", "[info ]", "missing -> run build_annotations.py"))
    return rows


def check_models() -> list:
    rows = []
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    rows.append(("HF_HOME", "[info ]", hf_home))
    rows.append(("HF_ENDPOINT", "[info ]", os.environ.get("HF_ENDPOINT", "(not set)")))
    bert_cache = Path(hf_home) / "hub"
    found = sorted(p.name for p in bert_cache.glob("models--*")) if bert_cache.is_dir() else []
    rows.append(("HF cached models", OK if found else "[info ]",
                 ", ".join(found[:6]) if found else "none (will download on first run)"))
    return rows


def main() -> int:
    setup_console()
    print("=" * 78)
    print("Environment check")
    print("=" * 78)
    print(f"python  : {sys.version.split()[0]}  ({platform.architecture()[0]})")
    print(f"platform: {platform.platform()}")
    print(f"exe     : {sys.executable}")
    print(f"project : {PROJECT_ROOT}")

    sections = [
        ("Python packages", check_python_packages()),
        ("PyTorch / CUDA", check_cuda()),
        ("Data", check_data()),
        ("HuggingFace cache", check_models()),
    ]
    for title, rows in sections:
        print("")
        print(f"--- {title} ---")
        for name, status, detail in rows:
            print(f"  {status} {name:18s} {detail}")

    print("")
    print("=" * 78)
    print("next steps:")
    print("  1) python src/build_annotations.py         # extract + build annotations.json")
    print("  2) python src/inspect_features.py          # dissect encoders (CPU ok)")
    print("  3) python src/train_image_only.py          # baseline A")
    print("  4) python src/train_text_only.py           # baseline B")
    print("  5) python src/ablation.py                  # 4 fusion variants")
    print("  6) python src/error_analysis.py --pred outputs/fusion_attention/test_predictions.npz")
    print("  7) python src/zeroshot_clip.py             # optional, CLIP zero-shot")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

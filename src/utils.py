"""utils.py —— 公共工具：随机种子、设备、配置合并、日志、混合精度。

约定
----
* 路径相关的工具统一放在 `paths.py`（纯标准库，不依赖 torch），
  这里把它们原样 re-export，所以老的 `from utils import project_path` 依然可用。
* 本文件里所有"相对路径"都相对于**项目根目录**（即 src/ 的上一级），
  因此无论你从哪个目录执行 `python src/xxx.py`，数据路径都不会错。
* 运行期日志统一用英文打印（避免 Windows 控制台 GBK 编码报错），
  代码注释与文档用中文。
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch
import yaml

from paths import (ZIP_ENV_VAR, ZIP_NAME, Logger, ensure_dir, find_dataset_zip,
                   project_path, setup_console, zip_search_report)

# re-export：其它模块习惯 from utils import PROJECT_ROOT
from paths import PROJECT_ROOT, REPO_ROOT  # noqa: F401


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer)
    if prefer.startswith("cuda"):
        print("[warn] CUDA is not available, falling back to CPU.")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        return f"cuda:{idx} ({name}, {total:.1f} GB)"
    return "cpu"


def read_yaml(path) -> dict:
    with open(project_path(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config 文件必须是字典（key: value），收到：{type(data)}")
    return data


def build_config(defaults: dict, config_path=None, cli_overrides: dict = None) -> dict:
    """合并三层配置：脚本内置默认值 < yaml 文件 < 命令行显式参数。

    命令行参数在 argparse 里都用 default=None，只有"用户真的写了"才会覆盖，
    这样 `--config configs/fusion_attention.yaml` 和 `--fusion early` 可以混用。
    """
    cfg = dict(defaults)
    if config_path:
        cfg.update(read_yaml(config_path))
    for key, value in (cli_overrides or {}).items():
        if value is not None:
            cfg[key] = value

    # 相对路径统一转绝对路径，避免 CWD 不同导致找不到数据
    for key in ("data_root", "ann_file", "out_dir", "resume", "viz_ckpt"):
        if key in cfg and cfg[key]:
            cfg[key] = str(project_path(cfg[key]))
    return cfg


def save_json(obj, path) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(project_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


class AverageMeter:
    """滑动平均：记录 loss / acc 这类逐 batch 指标。"""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def count_parameters(model: torch.nn.Module) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.start

    @property
    def text(self) -> str:
        return format_seconds(getattr(self, "elapsed", 0.0))


def make_grad_scaler(device: torch.device, enabled: bool = True):
    """兼容不同 torch 版本的 GradScaler 构造方式。"""
    enabled = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(device: torch.device, enabled: bool = True):
    enabled = bool(enabled and device.type == "cuda")
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)

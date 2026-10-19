"""paths.py —— 纯标准库的路径工具（**不 import torch**）。

为什么单独拆出来
----------------
`utils.py` 依赖 torch，而"检查环境"和"生成标注"这两步必须能在**装依赖之前**跑。
本文件只依赖 Python 标准库，因此可以被任何脚本安全导入：

    python src/check_env.py            # torch 没装也能跑，直接告诉你缺什么
    python src/build_annotations.py    # 只要有 zip 就能生成 annotations.json

路径约定
--------
    PROJECT_ROOT  = 代码工程目录（本目录，也是提交到 GitHub 时的**仓库根**，即 src/ 的上一级）
    PARENT_DIR    = 本目录的上一级（本地把 CUB_200_subset.zip 放在这里的 数据集/ 下）
    相对路径一律相对 PROJECT_ROOT 解析，因此从任何目录执行脚本都不会找不到文件。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent      # 本目录 = 仓库根
REPO_ROOT = PROJECT_ROOT                                   # 别名，语义更直观
PARENT_DIR = PROJECT_ROOT.parent                           # 数据集通常放在上一级

# 数据集压缩包：可放在多个习惯位置，也支持用环境变量指定
ZIP_NAME = "CUB_200_subset.zip"
ZIP_ENV_VAR = "CUB200_ZIP"
ZIP_CANDIDATES = (
    PROJECT_ROOT / "data" / ZIP_NAME,                      # 仓库内 data/（已被 .gitignore 忽略）
    PARENT_DIR / "数据集" / ZIP_NAME,                        # 本地默认位置（仓库外）
    PARENT_DIR / ZIP_NAME,                                 # 上一级目录根
    PROJECT_ROOT / ZIP_NAME,
    PROJECT_ROOT / "数据集" / ZIP_NAME,
    Path.cwd() / ZIP_NAME,
)


def setup_console() -> None:
    """把 stdout/stderr 切成 utf-8，避免 Windows 控制台中文乱码/报错。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def project_path(p) -> Path:
    """相对路径 -> PROJECT_ROOT 下的绝对路径；绝对路径原样返回。"""
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def ensure_dir(p) -> Path:
    path = project_path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_dataset_zip(explicit=None):
    """按优先级查找 CUB_200_subset.zip，找不到返回 None。

    顺序：命令行 --zip > 环境变量 CUB200_ZIP > data/ > 数据集/ > 仓库根目录 > 当前目录
    """
    if explicit:
        path = project_path(explicit)
        return path if path.is_file() else None
    env_value = os.environ.get(ZIP_ENV_VAR)
    if env_value:
        path = Path(env_value).expanduser()
        if path.is_file():
            return path
    for candidate in ZIP_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def zip_search_report() -> str:
    """把"我找过哪些位置"拼成一段提示，出错时直接打印给用户看。"""
    lines = [f"  - 环境变量 {ZIP_ENV_VAR}={os.environ.get(ZIP_ENV_VAR, '(未设置)')}"]
    lines += [f"  - {c}" for c in ZIP_CANDIDATES]
    lines.append(f"  - 或显式指定：python src/build_annotations.py --zip <你的路径>/{ZIP_NAME}")
    lines.append("  - 数据集不随仓库分发，获取方式见 docs/DATASET.md")
    return "\n".join(lines)


class Logger:
    """同时打印到终端并写文件（train.log），方便事后贴进报告。

    放在 paths.py 里是为了让"不需要 torch"的脚本（build_annotations / check_env）也能用。
    """

    def __init__(self, log_path=None):
        self.path = project_path(log_path) if log_path else None
        self.fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(self.path, "a", encoding="utf-8")
            self.fh.write("\n" + "=" * 80 + "\n")
            self.fh.write(f"run start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.fh.flush()

    def info(self, message: str = "") -> None:
        text = str(message)
        print(text, flush=True)
        if self.fh is not None:
            self.fh.write(text + "\n")
            self.fh.flush()

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None

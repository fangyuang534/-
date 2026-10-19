"""check_repo.py —— 仓库级的静态自检（CI 也跑这个）。

设计原则：**不 import torch / transformers**，只做静态检查，
所以任何人在 clone 之后、装依赖之前就能跑：

    python tests/check_repo.py

检查项
------
1. 语法：src/ 下所有 .py 能被编译（compile，不写 .pyc）
2. 配置：configs/*.yaml 的 key 必须都是对应训练脚本 DEFAULTS 里的 key
        （防止改了代码忘了改 yaml，或 yaml 里写错字段名被静默忽略）
3. 融合配置：proj_dim 必须能被 num_heads 整除
4. 跨模块 import：`from X import y` 里的 y 必须在 src/X.py 里真的定义过
        （模块被拆分/改名后最容易漏改这里，而 import 错误只有运行时才炸）
5. 可移植性：源码/配置里不允许出现本机绝对路径（C:\\...、/Users/xxx、Desktop 等）
6. 体积红线：不能有超过 50 MB 的待提交文件（GitHub 单文件上限 100 MB）
7. 忽略规则：.gitignore 必须覆盖 data/、.venv/、outputs/、*.zip、*.pt
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent      # 本目录就是 GitHub 仓库根
PROJECT_ROOT = REPO_ROOT                                # 代码同时在仓库根
SRC = REPO_ROOT / "src"
CONFIGS = REPO_ROOT / "configs"
MAX_BYTES = 50 * 1024 * 1024          # 待提交文件体积上限（50 MB）
TEXT_EXTS = {".py", ".yaml", ".yml", ".md", ".txt", ".sh", ".ps1"}

errors: list[str] = []
warnings: list[str] = []


def fail(message: str) -> None:
    errors.append(message)


def warn(message: str) -> None:
    warnings.append(message)


# ---------------------------------------------------------------- 1. 语法
def check_syntax() -> int:
    files = sorted(SRC.glob("*.py")) + sorted((REPO_ROOT / "tests").glob("*.py"))
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")   # 只编译、不写 .pyc（CI 里更干净）
        except SyntaxError as exc:
            fail(f"语法错误 {path.relative_to(REPO_ROOT)}:{exc.lineno}: {exc.msg}")
        except OSError as exc:
            fail(f"读取失败 {path.relative_to(REPO_ROOT)}: {exc}")
    return len(files)


# ---------------------------------------------------------------- 2/3. 配置
def defaults_of(script: Path, var: str = "DEFAULTS") -> dict:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == var for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{script.name} 里找不到 {var}")


def check_configs() -> list[tuple[str, int]]:
    """返回 [(配置文件, key 数量)]，同时检查 key 与类型。"""
    scripts = {
        "image_only": defaults_of(SRC / "train_image_only.py"),
        "text_only": defaults_of(SRC / "train_text_only.py"),
        "fusion": defaults_of(SRC / "train_multimodal.py"),
    }
    try:
        import yaml
    except ImportError:
        warn("未安装 PyYAML，跳过 configs/*.yaml 的字段检查（CI 里会装）")
        return []

    result = []
    for path in sorted(CONFIGS.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            fail(f"{path.name}: 顶层必须是 key: value 映射")
            continue
        if "image_only" in path.name:
            kind = "image_only"
        elif "text_only" in path.name:
            kind = "text_only"
        else:
            kind = "fusion"
        allowed = scripts[kind]
        unknown = sorted(set(cfg) - set(allowed))
        if unknown:
            fail(f"{path.name}: 未知字段 {unknown}（训练脚本不会读取它们）")
        for key, value in cfg.items():
            if key not in allowed:
                continue
            expect = allowed[key]
            if isinstance(expect, bool) and not isinstance(value, bool):
                fail(f"{path.name}: {key} 应为 bool，实际是 {value!r}")
            if isinstance(expect, (int, float)) and not isinstance(expect, bool) \
                    and isinstance(value, bool):
                fail(f"{path.name}: {key} 应为数字，实际是 {value!r}")
        if kind == "fusion":
            dim = int(cfg.get("proj_dim", 256))
            heads = int(cfg.get("num_heads", 8))
            if heads and dim % heads != 0:
                fail(f"{path.name}: proj_dim({dim}) 不能被 num_heads({heads}) 整除")
        result.append((path.name, len(cfg)))
    return result


# ---------------------------------------------------------------- 4. 跨模块 import
def top_level_names(tree: ast.Module) -> set[str]:
    """收集一个模块顶层"看得见"的名字：函数、类、赋值、import 进来的别名。"""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Tuple):
                    names.update(t.id for t in target.elts if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    return names


def check_imports() -> int:
    modules = {}
    trees = {}
    for path in sorted(SRC.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue                    # 语法错误已由 check_syntax 报告
        trees[path.stem] = tree
        modules[path.stem] = top_level_names(tree)

    checked = 0
    for mod_name, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level:
                continue                # 跳过相对 import（本工程不用）
            if not node.module or "." in node.module:
                continue
            if node.module not in modules:
                continue                # 第三方库
            available = modules[node.module]
            for alias in node.names:
                if alias.name == "*":
                    continue
                checked += 1
                if alias.name not in available:
                    fail(f"{mod_name}.py 从 {node.module} 导入的 "
                         f"`{alias.name}` 在 src/{node.module}.py 里不存在")
    return checked


# ---------------------------------------------------------------- 5. 可移植性
ABSOLUTE_PATTERNS = [
    re.compile(r"[A-Za-z]:\\\\?Users"),        # C:\Users / C:/Users
    re.compile(r"[A-Za-z]:\\\\?Desktop"),
    re.compile(r"/Users/[A-Za-z]"),            # macOS 家目录
    re.compile(r"/home/[A-Za-z]"),             # Linux 家目录
    re.compile(r"lenovo", re.IGNORECASE),      # 本机用户名
]


def check_portability() -> int:
    hits = 0
    for path in list(SRC.glob("*.py")) + list(CONFIGS.glob("*.yaml")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern in ABSOLUTE_PATTERNS:
                if pattern.search(line):
                    fail(f"硬编码本机路径 {path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()[:80]}")
                    hits += 1
                    break
    return hits


# ---------------------------------------------------------------- 5. 体积
def check_large_files() -> int:
    count = 0
    ignored_dirs = {"数据集", ".venv", "venv", "env", "__pycache__", "data",
                    "outputs", ".git", ".idea", ".vscode", "node_modules"}

    def on_error(_exc):
        return          # 某些目录（如 pip 临时目录）可能被占用，直接跳过

    for dirpath, dirnames, filenames in os.walk(REPO_ROOT, onerror=on_error):
        dirnames[:] = [d for d in dirnames
                       if d not in ignored_dirs and not d.startswith("pip-")]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_BYTES:
                fail(f"文件过大（{size / 1024 ** 2:.1f} MB），GitHub 会拒收："
                     f"{path.relative_to(REPO_ROOT)}")
                count += 1
    return count


# ---------------------------------------------------------------- 6. 忽略规则
def check_gitignore() -> None:
    path = REPO_ROOT / ".gitignore"
    if not path.is_file():
        fail("仓库根目录缺少 .gitignore")
        return
    text = path.read_text(encoding="utf-8")
    required = ["data/", "*.zip", ".venv/", "outputs/", "__pycache__/", "*.pt"]
    missing = [token for token in required if token not in text]
    if missing:
        fail(f".gitignore 缺少规则：{missing}")


# ---------------------------------------------------------------- 汇总
def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    print("=" * 72)
    print("repository check")
    print("=" * 72)
    print(f"repo root : {REPO_ROOT}")

    n_files = check_syntax()
    print(f"[1/7] syntax        : {n_files} python files checked")

    configs = check_configs()
    print(f"[2/7] configs       : {len(configs)} yaml checked -> "
          + ", ".join(f"{name}({n})" for name, n in configs))

    n_imports = check_imports()
    print(f"[3/7] imports       : {n_imports} cross-module names checked")

    hits = check_portability()
    print(f"[4/7] portability   : {hits} hardcoded absolute path(s)")

    big = check_large_files()
    print(f"[5/7] large files   : {big} file(s) over {MAX_BYTES // 1024 ** 2} MB")

    check_gitignore()
    print("[6/7] .gitignore    : checked")

    data_dir = PROJECT_ROOT / "data"
    print(f"[7/7] data dir      : {'present (ignored)' if data_dir.is_dir() else 'absent (ok)'}")

    print("-" * 72)
    for item in warnings:
        print(f"[warn] {item}")
    for item in errors:
        print(f"[FAIL] {item}")
    if errors:
        print(f"\n{len(errors)} problem(s) found.")
        return 1
    print("\nall checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# 鸟类多模态分类实验 · 详细运行手册

本仓库是**可直接运行的完整实验工程**：数据准备 → 单模态基线 → 模型拆解 → 四种融合 → 消融 → 零样本 CLIP → 错误分析。

- 仓库总览：[`../README.md`](../README.md)
- 数据集从哪来、放哪里：[`DATASET.md`](DATASET.md)
- 推到 GitHub 的步骤：[`GITHUB.md`](GITHUB.md)

---

## 0. 总览

```
鸟类多模态分类作业/          ← 仓库根（本目录）
├── README.md              ← 仓库首页（简介 / 快速开始）
├── docs/RUN.md            ← 本文件（详细运行手册）
├── docs/DATASET.md        ← 数据集说明
├── docs/GITHUB.md         ← 推送到 GitHub
├── .gitignore / .gitattributes / .editorconfig
├── tests/check_repo.py    ← 静态自检（CI 也跑它）
├── scripts/run_all.ps1|.sh← 一键复现
├── .github/workflows/ci.yml
├── requirements.txt
├── configs/               ← 6 个实验配置
└── src/                   ← 19 个源文件
```

## 0.1 源文件总览

```
src/                         ← 全部代码（19 个文件）
├── check_env.py             自检：环境/显卡/数据是否就绪（装依赖前就能跑）
├── build_annotations.py     解压数据 + 6:2:2 划分 + 生成 annotations.json
├── paths.py                 路径解析 / 数据集 zip 自动查找 / Logger
├── data.py                  数据集与 DataLoader
├── image_encoder.py         ResNet18 编码器（[B,3,224,224] → [B,512,7,7] → [B,49,512]）
├── text_encoder.py          BERT 编码器（tokenizer → embedding → [B,L,768] → 池化）
├── fusion.py                四种融合：early / late / attention / gated
├── model.py                 三个模型的组装 + 工厂
├── metrics.py               Accuracy / Top-5 / Macro P,R,F1 / 混淆矩阵
├── engine.py                训练与推理循环（AMP 混合精度）
├── trainer.py               通用训练流程（验证集选模型 → 测试集评估 → 存产物）
├── train_image_only.py      ① 单模态：Image-Only
├── train_text_only.py       ② 单模态：Text-Only
├── train_multimodal.py      ③ 多模态：--fusion early|late|attention|gated
├── ablation.py              ④ 一键跑 6 组对比，自动生成结果表
├── zeroshot_clip.py         ⑤ CLIP 零样本分类 + 图文检索（R@1/R@5）
├── inspect_features.py      ⑥ 模型拆解：逐层打印形状 + PCA 特征可视化
└── error_analysis.py        ⑦ 错误分析：混淆类别、错分样例、词级注意力
```

运行时会生成（均已被 `.gitignore` 排除）：`data/`（解压后的数据 + annotations.json）、`outputs/`（日志/权重/指标/图）。

**最短路径（照着复制粘贴即可）**

```bash
# 0) 在仓库根（本目录）执行
cd 鸟类多模态分类作业

# 1) 建虚拟环境并装依赖（只需一次，见第 1 节）
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2) 自检
python src/check_env.py

# 3) 数据准备（解压 + 生成标注，约 1~3 分钟）
python src/build_annotations.py

# 4) 单模态基线
python src/train_image_only.py
python src/train_text_only.py

# 5) 四种融合消融（依次跑 image_only/text_only/4 种融合，约 40~70 分钟）
python src/ablation.py

# 6) 拆解 / 错误分析 / 零样本
python src/inspect_features.py
python src/error_analysis.py --pred outputs/fusion_attention/test_predictions.npz \
    --viz_ckpt outputs/fusion_attention/checkpoints/best.pt --viz_samples 3
python src/zeroshot_clip.py
```

---

## 1. 环境准备（一次性）

### 1.1 依赖一览

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.10 ~ 3.13 | 建议 3.11 / 3.12 |
| PyTorch | ≥ 2.4 | 按 [官网](https://pytorch.org/get-started/locally/) 命令装对应 CUDA 版本 |
| torchvision | ≥ 0.19 | ResNet18 + ImageNet 权重 |
| transformers | ≥ 4.40 | BERT / CLIP |
| 其余 | Pillow、numpy、PyYAML、tqdm | 见 `requirements.txt` |
| 可选 | matplotlib、scikit-learn、pandas | 画图与线性探针，代码里都做了 try/except |

### 1.2 创建虚拟环境

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
#   （若报"禁止运行脚本"：Set-ExecutionPolicy -Scope CurrentUser RemoteSigned）
# Linux / macOS
source .venv/bin/activate
python -m pip install --upgrade pip
```

> 不想激活也行：Windows 用 `.\.venv\Scripts\python.exe src\xxx.py`，Linux 用 `.venv/bin/python src/xxx.py`。

### 1.3 安装 PyTorch（按显卡选 CUDA 版本）

```bash
# CUDA 12.6（RTX 30/40 系通用，驱动 ≥ 560）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 纯 CPU（能跑通，但很慢）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 1.4 安装其余依赖

```bash
pip install -r requirements.txt
```

### 1.5 下载预训练权重时的网络设置

需要下载的权重：

| 模型 | 用途 | 大小 |
|---|---|---|
| `resnet18`（torchvision） | 图像编码器 | ≈ 45 MB |
| `bert-base-uncased`（HuggingFace） | 文本编码器 | ≈ 440 MB |
| `openai/clip-vit-base-patch32` | 零样本实验（可选） | ≈ 600 MB |

HuggingFace 连不上时先设镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"      # Windows，仅当前终端
setx HF_ENDPOINT "https://hf-mirror.com"        # Windows，永久（需重开终端）
```
```bash
export HF_ENDPOINT=https://hf-mirror.com        # Linux / macOS
```

也可以直接用本地模型目录：`--text_model /path/to/bert-base-uncased`。
resnet 权重下载失败时，可先用随机初始化跑通流程（指标会下降）：`--image_pretrained 0`。

---

## 2. 数据准备

原始数据是 `CUB_200_subset.zip`（≈ 533 MB），**不在仓库里**。不用手动解压，脚本会处理，
并按下面的顺序自动查找压缩包：

1. `--zip <路径>`
2. 环境变量 `CUB200_ZIP`
3. `data/CUB_200_subset.zip`（仓库内，已被忽略）
4. `../数据集/CUB_200_subset.zip`（仓库外，本机默认位置）
5. `../CUB_200_subset.zip`
6. 当前目录等其它位置（详见 [`DATASET.md`](DATASET.md)）

```bash
python src/build_annotations.py
# 或显式指定：
python src/build_annotations.py --zip /path/to/CUB_200_subset.zip
```

脚本做 4 件事：

1. **解压**到 `data/cub200_subset/`；
2. **扫描图文对**：`images/<类别>/<名字>.jpg` 与 `text/<类别>/<名字>.txt` 按文件主名配对
   （注意：这份数据的描述**不在图片旁边**，而在与 `images/` 平级的 `text/` 目录里，脚本已适配）；
3. **按类别分层 6:2:2 划分**（固定 `seed=42`，可复现）；
4. 生成 `annotations.json`（含 `meta` / `classes` / `class_prompts` / `splits`），
   并额外导出 `annotations_train/val/test.json`（兼容课程参考脚本的数据接口）。

跑完会打印类似：

```
[scan] 100 classes
[scan] image-text pairs: 5864
[scan] images WITHOUT caption (dropped): 7
[split] train: 35xx images, 100 classes present
[split] val  : 11xx images, 100 classes present
[split] test : 11xx images, 100 classes present
```

> 具体数字以脚本输出为准（每类约 58~60 张，取整后 train 会略少于严格 60%）。
> 有 7 张图没有对应 txt（5864 个 txt vs 5871 张图），脚本会自动丢弃并记录。

换划分比例：

```bash
python src/build_annotations.py --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --out data/cub200_721
```

---

## 3. 各个实验怎么跑

> 所有脚本都在**本目录**下执行；`src/` 里的相对路径都按本目录解析，所以在哪个目录运行都不会找不到数据。
> 每个脚本都支持 `--config configs/xxx.yaml` 和命令行参数两种写法，**命令行参数优先级最高**。

### 3.1 ① Image-Only 基线

```bash
python src/train_image_only.py
# 等价于：python src/train_image_only.py --config configs/image_only.yaml
# 快速试跑：python src/train_image_only.py --epochs 3
```

### 3.2 ② Text-Only 基线

```bash
python src/train_text_only.py
python src/train_text_only.py --epochs 6              # 快速试跑
python src/train_text_only.py --text_pooling cls      # 换池化方式
```

产物在 `outputs/text_only/`。这一步在 CUB 上通常**很强**，原因是描述文本本身就把物种特征写清楚了
（"文本先验 / 描述泄漏"的问题值得在报告里讨论）。

### 3.3 ③ 多模态：四种融合

四种融合**共用同一套代码**，只改一个字段：

```bash
python src/train_multimodal.py --fusion early      # 早融合：concat + MLP
python src/train_multimodal.py --fusion late       # 晚融合：两个头 + 可学习权重 w
python src/train_multimodal.py --fusion attention  # 跨模态注意力（推荐）
python src/train_multimodal.py --fusion gated      # 门控融合
```

或直接用配置文件：

```bash
python src/train_multimodal.py --config configs/fusion_attention.yaml
```

默认配置对 8 GB 显存友好：ResNet18 微调 + **BERT 冻结**（只训练投影层/融合层/分类头），
每个 epoch 约 1 分钟。想再冲指标可以让 BERT 也参与微调：

```bash
python src/train_multimodal.py --fusion attention --freeze_text 0 --backbone_lr 2e-5 --epochs 5
```

### 3.4 ④ 一键消融

```bash
python src/ablation.py                             # 6 组全跑
python src/ablation.py --epochs 3                  # 3 个 epoch 快速验证流程
python src/ablation.py --variants attention gated  # 只跑指定的几组
python src/ablation.py --skip_baselines            # 只跑 4 种融合
```

产物：

| 文件 | 说明 |
|---|---|
| `outputs/ablation/results.csv` | 6 行数值（utf-8-sig，Excel 打开不乱码） |
| `outputs/ablation/results.md` | Markdown 表格，可直接贴进报告 |
| `outputs/ablation/<变体>/` | 每个变体自己的 log / metrics / 预测 |

### 3.5 ⑤ 模型拆解 / 特征分析

```bash
python src/inspect_features.py
python src/inspect_features.py --device cpu     # 没有 GPU 也能跑
python src/inspect_features.py --num_plot 500   # PCA 图用更多样本
```

终端（以及 `outputs/inspect/features.log`）会按 6 段打印：

1. Tokenizer 如何把句子变成 id
2. Embedding 三者相加（Token + Segment + Position）
3. `[B,L,768]` 如何池化成句向量、**mask 关掉会差多少**
4. 图像 `[B,3,224,224] → [B,512,7,7] → [B,49,512] / [B,512]`
5. 同模态相似度矩阵
6. PCA 散点图 → `outputs/inspect/pca_image.png`、`pca_text.png`

### 3.6 ⑥ 错误分析

```bash
# 基础分析（只读预测结果，不重新训练）
python src/error_analysis.py --pred outputs/fusion_attention/test_predictions.npz

# 加上"词级注意力可视化"（需要指定该次训练的权重）
python src/error_analysis.py --pred outputs/fusion_attention/test_predictions.npz \
    --viz_ckpt outputs/fusion_attention/checkpoints/best.pt --viz_samples 3
```

产物在 `outputs/error_analysis/`：`top_confusions.csv`、`worst_classes.csv|png`、
`misclassified.png`、`gate_distribution.png`、`attention/attn_*.png`。

### 3.7 ⑦ CLIP 零样本 + 图文检索（可选）

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内建议先设镜像（Windows 用 $env:）
python src/zeroshot_clip.py
python src/zeroshot_clip.py --limit 300    # 只用 300 张测试图，快速看效果
python src/zeroshot_clip.py --linear_probe # 额外做"CLIP 特征 + 逻辑回归"线性探针
```

输出：零样本 Accuracy / Macro-F1，以及检索 `i2t_R@1`、`i2t_R@5`、`t2i_R@1`、`t2i_R@5`。
需要下载 CLIP 权重（约 600 MB），不跑也不影响前面所有实验。

---

## 4. 产物一览（每个实验都一样）

以 `outputs/fusion_attention/` 为例：

| 文件 | 内容 |
|---|---|
| `train.log` | 完整日志：每 epoch 的 loss / train_acc / val_acc / val_f1 / 耗时 |
| `checkpoints/best.pt` | 验证集准确率最高的权重 |
| `metrics.json` | 测试集 Accuracy / Top-5 / Macro P,R,F1 / 100 类逐类准确率 |
| `history.json` | 训练曲线数据（画 loss/acc 曲线） |
| `test_predictions.npz` | 测试集预测（y_true / y_pred / probs / paths / gate） |
| `config.json` | 本次运行的完整配置（复现用） |
| `result.json` | 一行汇总（消融表就取这个） |

---

## 5. 预计耗时（参考值）

以下是 **RTX 4060 Laptop 8 GB** 上的经验值，仅供估计，不同机器差异很大：

| 实验 | 默认 epoch | 预计耗时 |
|---|---|---|
| `build_annotations.py` | — | 1~3 分钟（主要是解压） |
| `train_image_only.py` | 15 | 10~15 分钟 |
| `train_text_only.py` | 12 | 8~12 分钟 |
| `train_multimodal.py`（单组，BERT 冻结） | 8 | 6~10 分钟 |
| `ablation.py`（6 组全跑） | 8 | 45~70 分钟 |
| `error_analysis.py` | — | 1~2 分钟 |
| `inspect_features.py` | — | 2~4 分钟 |
| `zeroshot_clip.py` | — | 3~6 分钟（不含下载） |

没有 GPU 也能跑：脚本会自动退回 CPU（也可 `--device cpu` 强制），
但建议把 `--epochs` 降到 2~3、加 `--freeze_image 1`、把 `--image_size` 降到 128，否则很慢。

---

## 6. 常见问题（FAQ）

**Q1：`CUDA out of memory`**
把 batch 调小：`--batch_size 16`（注意力融合最吃显存），或 `--freeze_image 1` 冻结 ResNet 主干。

**Q2：`OSError: We couldn't connect to 'https://huggingface.co'`**
设镜像后重跑：`$env:HF_ENDPOINT = "https://hf-mirror.com"`；
或用已下好的本地目录：`--text_model D:/models/bert-base-uncased`。

**Q3：下载 resnet 权重失败（`URLError` / 超时）**
`--image_pretrained 0` 用随机初始化先跑通（指标会掉一些，说明里注明即可）。

**Q4：Windows 上 `DataLoader` 卡住 / `worker exited unexpectedly`**
`--num_workers 0`（最稳）。

**Q5：`找不到数据集压缩包` / `找不到标注文件`**
前者看 `docs/DATASET.md` 的放置位置，或用 `--zip` 指定；
后者先跑 `python src/build_annotations.py`。

**Q6：终端中文乱码**
所有运行日志都是英文打印（避免 Windows GBK 编码问题），中文只出现在注释、yaml 注释和文档里。

**Q7：指标很低（Acc < 20%）**
检查：① 是否漏了 `--image_pretrained 1`；② BERT 是否加载成功（日志里 `[model] total params` 应约 110 M）；
③ `--lr` 是否被改得过大；④ 数据是否完整（`[scan] image-text pairs` 应约 5864）。

**Q8：想复现同一个结果**
所有脚本固定 `seed=42`（`--seed` 可改），数据划分也固定；但 GPU 上 cudnn 非确定性会带来约 ±0.5% 波动，属正常。

**Q9：能把课程提供的参考脚本（`02_singleModal_TextOnly.py`，不在本仓库）直接跑起来吗？**
可以：`build_annotations.py` 已导出它需要的 `annotations_train.json / annotations_val.json / annotations_test.json`，
把它的 `root_dir` 改成本项目的 `data/cub200_subset` 即可。

---

## 7. 代码与实验手册章节对照

| 手册章节 | 本工程对应 | 说明 |
|---|---|---|
| §1.2 安装依赖 | `requirements.txt` / 第 1 节 | — |
| §1.3 数据准备 | `src/build_annotations.py` | 6:2:2 + annotations.json（txt 与 jpg 按文件名配对） |
| §2.1 Image-Only | `src/train_image_only.py` | ResNet18，训练/测试/指标齐全 |
| §2.2 Text-Only | `src/train_text_only.py` | BERT + mean pooling（改进：划分固定、加验证集、缓存分词、分层学习率） |
| §2 "把模型拆开看特征" | `src/inspect_features.py` | 逐层形状 + 池化对比 + PCA 可视化 |
| §3 Image Encoder | `src/image_encoder.py` | `[B,512,7,7] → [B,49,512]`，`flatten(2).transpose(1,2)` |
| §3 Text Encoder | `src/text_encoder.py` | tokenizer → embedding → transformer → `cls`/`mean` 池化（含 mask） |
| §4.1/4.2 四种融合 | `src/fusion.py` | 四类接口一致，返回 dict（含 logits / gate / attn_weights） |
| §4 消融实验 | `src/ablation.py` | 一键 6 组 + 自动出表 |
| §5 CLIP 思想实验 | `src/zeroshot_clip.py` | 零样本 Acc + R@1/R@5 检索 |
| §6/§7 分析与讨论 | `src/error_analysis.py` | 混淆类别、错分样例、词级注意力、门控分布 |

---

## 8. 运行会生成什么（可以放心删）

| 目录/文件 | 大小 | 说明 |
|---|---|---|
| `data/cub200_subset/` | 约 600 MB | 解压数据 + annotations.json（删了重跑 `build_annotations.py`） |
| `outputs/` | 每组 10~60 MB | 日志、权重、指标、图；删掉不影响代码 |
| `.venv/` | 约 3 GB | 虚拟环境（已在 `.gitignore` 里） |
| `~/.cache/huggingface` | 约 1 GB | BERT / CLIP 权重缓存 |

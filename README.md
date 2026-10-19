# 基于图像-文本联合表示的鸟类多模态分类

> 计算机视觉课程作业 · CUB_200_subset（100 类 / 5864 组图文对）
> ResNet18 + BERT → 四种融合（Early / Late / Cross-Attention / Gated）→ 消融 · 零样本 CLIP · 错误分析

![python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![pytorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)
![transformers](https://img.shields.io/badge/Transformers-4.40%2B-FFD21E)
![ci](https://img.shields.io/badge/ci-static%20check-informational)

<!-- 拥有仓库后，可把上面最后一行换成官方 CI 徽章：
![ci](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg)
-->

---

## 这是什么

一个**完整可复现**的多模态细粒度分类工程：从原始 `jpg + txt` 出发，一直做到融合消融、零样本分类、
注意力可视化与错误分析。代码按"单模态 → 双模态 → 跨模态对齐 → 多模态融合 → 消融与错误分析"组织，
每一步都能单独运行、单独评估。

| 模块 | 实现要点 |
|---|---|
| 图像编码器 | ResNet18 去掉 avgpool+fc → `[B,512,7,7]` → 池化 `[B,512]` / 展平 `[B,49,512]`（49 个"视觉词"） |
| 文本编码器 | BERT tokenizer → embedding（Token+Position+Segment）→ 12 层 Transformer `[B,L,768]` → CLS / masked-mean 池化 |
| 融合 | Early（concat+MLP）、Late（双头 + 可学习权重）、Cross-Attention（图像 token 作 Q）、Gated（`a·img+(1-a)·txt`） |
| 评估 | Accuracy / Top-5 / Macro-P,R,F1 / 每类准确率 / 混淆矩阵（纯 numpy 实现） |
| 分析 | 门控权重分布、词级注意力图、PCA 特征可视化、错分样例网格 |
| 零样本 | CLIP 提示词分类 + 图文检索 R@1 / R@5 + 可选线性探针 |

## 目录结构

```
鸟类多模态分类作业/            ← 本仓库根
├── README.md                本文件
├── requirements.txt
├── .gitignore / .gitattributes / .editorconfig
├── configs/                 6 个实验配置（改一个字段就换实验）
│   ├── image_only.yaml      text_only.yaml
│   └── fusion_early.yaml    fusion_late.yaml    fusion_attention.yaml    fusion_gated.yaml
├── src/                     19 个源文件
├── docs/
│   ├── RUN.md               详细运行手册（环境、命令、耗时、9 条 FAQ）
│   ├── DATASET.md           数据集从哪来、放哪里、怎么生成标注
│   └── GITHUB.md            如何推到 GitHub（含报错处理）
├── tests/check_repo.py      静态自检（不装 torch 也能跑，CI 跑的就是它）
├── scripts/
│   ├── run_all.ps1          Windows 一键复现
│   └── run_all.sh           Linux / macOS 一键复现
├── .github/workflows/ci.yml 每次 push 自动跑静态自检
├── data/                    运行时生成（解压数据 + annotations.json）· 已在 .gitignore
└── outputs/                 运行时生成（日志/权重/指标/图）· 已在 .gitignore
```

`src/` 一览：

| 文件 | 作用 |
|---|---|
| `paths.py` | 路径解析 + 数据集 zip 自动查找 + Logger（**纯标准库，无 torch 依赖**） |
| `check_env.py` | 环境自检：包、GPU、数据、HF 缓存（装依赖前就能跑） |
| `build_annotations.py` | 解压 → 图文配对 → 按类分层 6:2:2 → `annotations.json` |
| `data.py` | Dataset / DataLoader；训练时随机采一条 caption，分词结果缓存 |
| `image_encoder.py` | ResNet18 图像编码器（`forward_with_maps` 便于"拆开看"） |
| `text_encoder.py` | BERT 文本编码器（`cls` / `mean` 两种池化，带 mask） |
| `fusion.py` | 四种融合模块，接口统一，附带 attention / gate 供可视化 |
| `model.py` | 三个模型的组装 + `build_model` 工厂 + 参数量统计 |
| `metrics.py` | Accuracy / Top-5 / Macro P,R,F1 / 每类准确率 / 混淆矩阵 |
| `engine.py` | 训练与推理循环（AMP 混合精度、梯度裁剪、结果落盘） |
| `trainer.py` | 通用训练流程：验证集选最优 → 测试集评估一次 → 保存日志/权重/预测 |
| `train_image_only.py` | 基线 A：Image-Only |
| `train_text_only.py` | 基线 B：Text-Only |
| `train_multimodal.py` | 多模态：`--fusion early\|late\|attention\|gated` |
| `ablation.py` | 一键跑 6 组对比，自动生成 `results.csv` / `results.md` |
| `zeroshot_clip.py` | CLIP 零样本分类 + 图文检索 R@1/R@5（可选） |
| `inspect_features.py` | 模型拆解：逐层形状、mask 对池化的影响、PCA 特征可视化 |
| `error_analysis.py` | 混淆类别对、最难类、错分样例、门控分布、词级注意力图 |

## 快速开始

> 详细版（三平台命令、逐实验说明、耗时预估、9 条 FAQ）见 **[docs/RUN.md](docs/RUN.md)**。

```bash
# 1) 环境（一次性）
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1     Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # 纯 CPU 用 /whl/cpu
pip install -r requirements.txt
# 国内下载 BERT/CLIP 时可设镜像：export HF_ENDPOINT=https://hf-mirror.com

# 2) 自检（此时数据集可以还没就位）
python src/check_env.py

# 3) 数据准备：自动查找 CUB_200_subset.zip → 解压 → 生成 annotations.json
python src/build_annotations.py

# 4) 单模态基线
python src/train_image_only.py
python src/train_text_only.py

# 5) 四种融合 + 消融（生成对比表）
python src/ablation.py

# 6) 拆解 / 错误分析 / 零样本
python src/inspect_features.py
python src/error_analysis.py --pred outputs/fusion_attention/test_predictions.npz \
    --viz_ckpt outputs/fusion_attention/checkpoints/best.pt
python src/zeroshot_clip.py
```

或者一条命令跑完：

```powershell
.\scripts\run_all.ps1                 # Windows
bash scripts/run_all.sh               # Linux / macOS
```

## 数据

**数据集不在本仓库里**（533 MB，且 CUB 有学术使用条款）。把 `CUB_200_subset.zip` 放到
`data/`、或仓库上一级的 `数据集/`、或用环境变量指定，脚本会自动找到：

```powershell
$env:CUB200_ZIP = "D:\datasets\CUB_200_subset.zip"   # Windows
export CUB200_ZIP=/data/CUB_200_subset.zip           # Linux/macOS
```

目录结构、配对规则、"只有官方完整数据集时如何裁剪出这 100 类子集"，见 **[docs/DATASET.md](docs/DATASET.md)**。

## 实验结果

> 数字由脚本自动写入 `outputs/`，完整版在 `outputs/ablation/results.md`。
> **本仓库未附带实测数值**（实验在本地 GPU 上按上面的命令复现），复现后把数字填进来即可。

| 变体 | 模态 | 融合方式 | Test Acc | Top-5 | Macro-F1 | 可训练参数 |
|---|---|---|---|---|---|---|
| `image_only` | 图像 | — | — | — | — | ≈11.2 M |
| `text_only` | 文本 | — | — | — | — | ≈109.5 M |
| `fusion_early` | 图文 | Early（concat+MLP） | — | — | — | ≈11.6 M |
| `fusion_late` | 图文 | Late（双头加权） | — | — | — | ≈11.7 M |
| `fusion_attention` | 图文 | Cross-Attention | — | — | — | ≈12.4 M |
| `fusion_gated` | 图文 | Gated | — | — | — | ≈11.7 M |

## 实验设置

| 项目 | 设置 |
|---|---|
| 输入 | 224×224，ImageNet 标准化；文本 `max_length=48` |
| 优化 | AdamW + cosine 退火，label smoothing 0.1，dropout 0.2，梯度裁剪 1.0，AMP fp16 |
| 训练策略 | 每 epoch 验证一次，只保留验证集最优权重，**测试集只评估一次** |
| 主干 | ResNet18 微调；BERT-base 默认**冻结**（`--freeze_text 0` 可解冻，配 `backbone_lr 2e-5`） |
| 融合层 | 两模态先投影到 `proj_dim=256` + LayerNorm，再做融合 |
| 复现性 | 固定数据划分（`seed=42`）+ 固定初始化种子 |

## 自检与 CI

```bash
python tests/check_repo.py
```

不装 torch 也能跑，检查：19 个 py 文件语法、6 个 yaml 的字段是否与训练脚本 `DEFAULTS` 一致、
124 处跨模块 import 名是否真实存在、是否混入本机绝对路径、是否有超过 50 MB 的待提交文件、
`.gitignore` 是否覆盖数据与产物。`.github/workflows/ci.yml` 在每次 push/PR 时自动执行。

## 说明与致谢

- 数据集：Caltech-UCSD Birds-200-2011（CUB），学术用途，引用见 [docs/DATASET.md](docs/DATASET.md)。
- 预训练权重：torchvision ResNet18（ImageNet）、HuggingFace `bert-base-uncased`、`openai/clip-vit-base-patch32`。
- 课程提供的实验手册与参考脚本仅作对照，未包含在本仓库中。
- 本仓库未附带 LICENSE 文件；如需开源许可，可自行添加 MIT / Apache-2.0。

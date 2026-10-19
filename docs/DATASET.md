# 数据集说明（CUB_200_subset）

> **数据集不随本仓库分发。** 原因有两个：体积（解压后约 600 MB）以及 CUB 数据集的学术使用条款。
> 仓库里只有代码；数据放在本地，脚本会自动找到它（见下文“放置位置”）。

---

## 1. 数据集是什么

- 来源：**Caltech-UCSD Birds-200-2011**（CUB-200-2011），细粒度鸟类图像数据集。
- 本实验用的是它的 **100 类子集**（`001.Black_footed_Albatross` … `100.Brown_Pelican`），
  每张图片额外配 10 条英文描述（由标注者撰写）。

| 项目 | 数值 |
|---|---|
| 类别数 | 100 |
| 图片数 | 5871（`.jpg`） |
| 描述文件数 | 5864（`.txt`，每个 10 行） |
| 可用图文对 | 5864（有 7 张图没有对应描述，脚本自动丢弃） |
| 描述长度 | 约 10~25 个英文词 |
| 划分 | 6 : 2 : 2，按类别分层，`seed = 42` |

## 2. 目录结构

zip 解压后的结构与**图片描述分开放**（这点和很多教程里的写法不同，脚本已适配）：

```
CUB_200_subset/
├── images/
│   ├── 001.Black_footed_Albatross/
│   │   ├── Black_Footed_Albatross_0001_796111.jpg
│   │   └── ...
│   ├── 002.Laysan_Albatross/
│   └── ...                                  100 个类别目录
└── text/                                    ← 与 images/ 平级
    ├── 001.Black_footed_Albatross/
    │   ├── Black_Footed_Albatross_0001_796111.txt   ← 主名与 jpg 相同
    │   └── ...
    └── ...                                  100 个类别目录
```

配对规则：**同一类别目录下、文件主名（不含后缀）相同的 `.jpg` 与 `.txt` 组成一个样本。**

一个 `.txt` 的内容（每行一条描述，共 10 行）：

```
the medium sized bird has a dark grey color and long wings.
the bird is dark grey brown with a thick curved bill.
bird has brown body feathers, white breast feathers.
...
```

## 3. 放在哪里（任选其一）

> **数据在仓库外面**：本目录（`鸟类多模态分类作业/`）就是仓库根，数据集放在它的上一级 `数据集/`
> 或本地的 `data/` 里，两者都被 `.gitignore` 排除，不会被提交。

脚本 `src/build_annotations.py` 会按下面的顺序自动查找 `CUB_200_subset.zip`：

1. 命令行参数：`--zip <路径>`
2. 环境变量：`CUB200_ZIP`
3. `data/CUB_200_subset.zip`（仓库内，已被忽略）
4. `../数据集/CUB_200_subset.zip` ← **推荐，本机就是放在这里**（仓库外）
5. `../CUB_200_subset.zip`
6. `./CUB_200_subset.zip`
7. `./数据集/CUB_200_subset.zip`
8. 当前工作目录下的 `CUB_200_subset.zip`

Windows PowerShell 用环境变量指定：

```powershell
$env:CUB200_ZIP = "D:\datasets\CUB_200_subset.zip"
```

Linux / macOS：

```bash
export CUB200_ZIP=/data/CUB_200_subset.zip
```

## 4. 生成标注文件

```bash
# 在本目录（仓库根）下执行
python src/build_annotations.py
```

它会做四件事：

1. 解压到 `data/cub200_subset/`（已解压则跳过）；
2. 扫描 `images/` 与 `text/`，按文件名配对；
3. 在每个类别内部按 6:2:2 分层切分（固定 seed）；
4. 写出 `data/cub200_subset/annotations.json`，并额外导出
   `annotations_train.json / annotations_val.json / annotations_test.json`（兼容旧脚本格式）。

成功时日志大致是：

```
[scan] 100 classes
[scan] image-text pairs: 5864
[scan] images WITHOUT caption (dropped): 7
[split] train: 35xx images, 100 classes present
[split] val  : 11xx images, 100 classes present
[split] test : 11xx images, 100 classes present
```

> 具体数字以脚本实际打印为准：因为每类样本数（约 58~60）不能整除，
> `int(round(0.6n))` 取整后 train 会略少于严格的 60%。

## 5. 没有这份 100 类子集怎么办

两个办法：

### 办法 A：向课程/老师获取

课程通常直接提供 `CUB_200_subset.zip`（就是上面这种 `images/` + `text/` 结构）。

### 办法 B：用官方 CUB-200-2011 自己构造

1. 从官方地址下载 CUB-200-2011（`CUB_200_2011.tgz` + `text_c10`，官方页面有说明）；
2. 只保留前 100 个类别（`001`~`100`），拼成如下结构：

```
数据集/CUB_200_subset/
├── images/
│   └── 001.Black_footed_Albatross/...
└── text/                     ← 即官方的 text_c10/ 重命名
    └── 001.Black_footed_Albatross/...
```

3. 因为已经是解压状态，用 `--skip_extract` 直接生成标注：

```bash
python src/build_annotations.py --skip_extract --out data/cub200_subset
```

（`--out` 指向你手工整理好的目录，脚本会直接扫描它并写 `annotations.json`。
也可以把它放到上一级 `../数据集/CUB_200_subset` 再 `--out ../数据集/CUB_200_subset`。）

## 6. 自检清单

跑完 `build_annotations.py` 后，可以这样确认数据准备是否正确：

```bash
python src/check_env.py
```

应当看到：

```
[ OK ] dataset zip        ...\数据集\CUB_200_subset.zip  (533 MB)
[ OK ] data/images        100 class folders under ...\data\cub200_subset\images
[ OK ] annotations.json   classes=100 train/val/test=35xx/11xx/11xx
```

以及 `data/cub200_subset/annotations.json` 里：

- `meta.num_classes == 100`
- `meta.num_images ≈ 5864`
- `splits` 三个子集都包含 100 个类别

## 7. 引用

```bibtex
@techreport{cub200-2011,
  title       = {The Caltech-UCSD Birds-200-2011 Dataset},
  author      = {C. Wah and S. Branson and P. Welinder and P. Perona and S. Belongie},
  institution = {California Institute of Technology},
  year        = {2011}
}
```

# 把这份作业提交到 GitHub

> 现状（本机）：
> - **没有安装 Git 命令行**（`git` 不在 PATH 里）；
> - 桌面上有 **GitHub Desktop**，它自带 Git，最省事。
> - **仓库根就是本目录**（`鸟类多模态分类作业/`）——只提交它，上一级的 `原/`（课程资料）和 `数据集/`（533 MB）都不进仓库。
> - `.gitignore`、`.gitattributes`、`README.md`、`docs/`、`tests/`、`.github/`、`scripts/` 已经就位。

---

## 0. 先确认这 4 件事（30 秒）

| 检查 | 期望 |
|---|---|
| `data/`、`outputs/` 是否被忽略 | 提交列表里**看不到**（有就说明 `.gitignore` 没生效） |
| `.venv/` 是否被忽略 | 看不到（约 3 GB） |
| 有没有 `.zip`/`.pt`/`.npz` 混进来 | 没有（数据集在仓库外，本来就进不来） |
| 有没有别的 >50 MB 文件 | 没有。跑一次 `python tests/check_repo.py` 就知道 |

---

## 1. 方案 A：GitHub Desktop（推荐，图形化）

1. 打开 **GitHub Desktop** → `File` → `Add local repository…` → 选中**本目录**（`鸟类多模态分类作业`）
   - 提示"这不是 Git 仓库"时点 **`create a repository`**，确认三项：
     - `Name`：建议**英文**，如 `bird-multimodal-classification`（中文仓库名在 URL 里会被转义成一串 `%E9...`）
     - `Local path`：指向**本目录**（有 `README.md`、`src/`、`configs/` 的那一层，**不要再往下钻**）
     - **不要**勾选 "Initialize with README / .gitignore / License"（会覆盖已有的）
   - 点 `Create repository`
2. 左侧 `Changes` 面板检查文件列表，应该是：
   ```
   .github/workflows/ci.yml     .gitattributes     .editorconfig
   .gitignore                   README.md          requirements.txt
   configs/*.yaml               docs/*.md          scripts/*          src/*.py
   tests/check_repo.py
   ```
   **不应该**出现 `data/`、`outputs/`、`.venv/`、`pip-*`、任何 `.zip`/`.pt`。
3. 左下 `Summary` 填提交信息，然后 `Commit to main`：
   ```
   feat: 鸟类多模态分类实验（数据准备/单模态基线/四种融合/消融/零样本/错误分析）
   ```
4. 点上方 **`Publish repository`**：取消勾选 `Keep this code private`（除非课程要求私有）→ `Publish repository`。

> 之后每次改代码：`Changes` → 写 summary → `Commit to main` → `Push origin`。

---

## 2. 方案 B：Git 命令行（要先装 Git）

1. 装 **Git for Windows**：<https://git-scm.com/download/win>（默认选项即可），装完**重开终端**：
   ```powershell
   git --version
   ```
2. 首次使用配置身份：
   ```powershell
   git config --global user.name  "你的名字"
   git config --global user.email "你的邮箱"
   ```
3. 在**本目录**（仓库根）初始化并提交：
   ```powershell
   cd "D:\你的路径\鸟类多模态分类作业"    # 换成你自己的路径

   git init -b main
   git add .
   git status --short          # 重点看：不该出现 data/、outputs/、.venv/、*.zip

   git commit -m "feat: 鸟类多模态分类实验（数据准备/单模态基线/四种融合/消融/零样本/错误分析）"
   ```
4. 在 GitHub 网页新建**空仓库**（不要勾选 README/.gitignore/LICENSE），然后：
   ```powershell
   git remote add origin https://github.com/<你的用户名>/<仓库名>.git
   git push -u origin main
   ```
5. 日常：
   ```powershell
   git add .
   git commit -m "docs: 补充实验结果"
   git push
   ```

> 推送时 GitHub 不接受账号密码，需要 **Personal Access Token**：
> `Settings` → `Developer settings` → `Personal access tokens` → `Tokens (classic)` → 勾 `repo` → 生成，
> 把 token 当密码粘贴。或者直接用 GitHub Desktop / GitHub CLI 登录。

---

## 3. 万一误提交了大文件

先看加了什么：

```powershell
git status --short
git ls-files | Measure-Object -Line
```

撤销跟踪（**不删除本地文件**）：

```powershell
git rm -r --cached data outputs
git commit -m "chore: 不再跟踪数据与训练产物"
```

已经 push 且体积很大时，需要改写历史（谨慎）：

```powershell
pip install git-filter-repo
git filter-repo --path data --path outputs --invert-paths
git push --force
```

---

## 4. 常见报错

| 报错 | 原因 | 处理 |
|---|---|---|
| `remote: error: GH001: Large files detected` | 有 >100 MB 的文件被提交 | 按第 3 节撤销跟踪后重提 |
| `RPC failed; HTTP 413` | 单次推送太大 | 同上；或 `git config --global http.postBuffer 524288000` |
| `fatal: not a git repository` | 不在仓库根 | `cd` 到 `鸟类多模态分类作业` 再执行（**不要**在上一级执行） |
| `LF will be replaced by CRLF` | Windows 换行提示 | 无害，忽略；`.gitattributes` 已统一为 LF |
| 中文文件名乱码 | 终端编码 | GitHub 网页显示正常；本地可 `git config --global core.quotepath false` |
| `Permission denied (publickey)` | 用了 SSH 地址但没配密钥 | 改 HTTPS + token，或配置 SSH key |
| `src/paths.py` 中文注释变乱码 | 编辑器按 GBK 打开 | 项目统一 UTF-8；VS Code 右下角选 UTF-8 保存 |

---

## 5. 提交前跑一次自检

```powershell
python tests/check_repo.py
```

不装 torch 的前提下检查：语法、yaml 字段、跨模块 import 名、本机绝对路径、>50 MB 文件、`.gitignore` 覆盖。
全部通过会打印 `all checks passed.`；CI（`.github/workflows/ci.yml`）每次 push 也会跑同一脚本。

---

## 6. 提交前的人工清单

- [ ] `git status` 里没有 `data/`、`outputs/`、`.venv/`、`pip-*`
- [ ] README 里的徽章链接如需换成真实 CI 徽章：把 `README.md` 顶部注释里的
      `![ci](https://github.com/<OWNER>/<REPO>/...)` 替换掉占位徽章
- [ ] 不要在仓库里写学号 + 手机号等敏感信息（需要署名就只写名字）
- [ ] 仓库页右上角 `About` 补一句描述 + Topics（`multimodal-learning`、`clip`、`cub-200-2011`、`pytorch`）
- [ ] 课程要求私有仓库时：`Settings` → `General` → `Danger Zone` → `Change visibility` → Private

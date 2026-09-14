# ICT 图片 15 分类器

把 ICT 领域的图片自动分到 **15 个细类**（工程图纸、波形图、器件照片、认证文档等）。

基于预训练视觉编码器 + 轻量分类头：先做**冻结特征 backbone 大比拼**选出最合适的底座，再解冻最后若干层微调。

---

## 目录

- [项目范围](#项目范围)
- [数据实况](#数据实况)
- [技术路线](#技术路线)
- [环境准备](#环境准备)
- [模型权重](#模型权重)
- [快速开始](#快速开始)
- [单步详解](#单步详解)
- [多 GPU 运行](#多-gpu-运行)
- [指标怎么读](#指标怎么读)
- [目录结构](#目录结构)
- [已知问题与排查](#已知问题与排查)
- [下一步方向](#下一步方向)

---

## 项目范围

| | |
|---|---|
| **做** | 15 类单标签分类。输出：最可能类别 + 置信度 + top-3 |
| **不做** | **不做相关性门控、不做拒识**。`不相关` 从数据划分开始就排除，当作不存在。无关图片过滤由外部环节负责 |

数据源：`multi_data/data0910/rejudged/`，一层子目录名即类别名。

---

## 数据实况

排除 `不相关` 后共 **11,529 张 / 15 类**，全部可读、0 张损坏。

| 类别 | train | val | 合计 | 备注 |
|---|---:|---:|---:|---|
| 电路与工程设计图 | 3701 | 925 | 4626 | 占 40% |
| 产品与结构图片 | 1428 | 364 | 1792 | |
| 信号与波形图片 | 995 | 253 | 1248 | |
| 文档与认证图片 | 856 | 209 | 1065 | |
| PCB与电路板图片 | 408 | 104 | 512 | |
| 测试与检测图片 | 376 | 82 | 458 | |
| 电子元器件图片 | 321 | 85 | 406 | |
| 仿真分析图片 | 316 | 82 | 398 | |
| 工程实验图片 | 247 | 61 | 308 | |
| 制造工艺图片 | 173 | 43 | 216 | |
| 器件资料 | 144 | 38 | 182 | unreliable |
| 安装部署与运维图片 | 116 | 31 | 147 | unreliable |
| 故障与维修图片 | 91 | 24 | 115 | unreliable |
| 包装运输与仓储图片 | 36 | 10 | 46 | unreliable |
| 质量类 | 8 | 2 | 10 | unreliable，val 仅 2 张 |
| **合计** | **9216** | **2313** | **11529** | |

三件必须先知道的事：

1. **极度不均衡**：`电路与工程设计图 : 质量类 = 3701 : 8`，约 **463 : 1**。这决定了必须用类别权重，且**绝不能用 accuracy 或 loss 选模型**。
2. **5 个类标了 `unreliable`**：样本太少，或类内一致性无保证。它们照常参与训练，但**指标不进主口径**——所以评估报两个 macro-F1。
3. **有近重复**：md5 层面 0 重复，但 dHash 层面存在同图不同压缩的变体。划分阶段做了近重复分组，**组整体分配**，保证同组不跨 train/val。顺带查出 **87 个近重复组横跨两个不同标签**——同一张图在图库里有两个标，是标签噪声的直接证据。

内容形态是**混合**的：工程图纸、CAE/CAD 软件截图、论文插图、设备实拍、灰度 X-ray/CT 检测图、认证文档。**判别信号主要是图里的文字标签与排版，不是自然图像纹理**。这直接决定了下面两个预处理选择。

---

## 技术路线

```
prepare_data.py   →  划分数据 + 近重复防泄漏
       ↓
extract_features.py  →  6 个 backbone 抽冻结特征缓存
       ↓
linear_probe.py   →  backbone 大比拼，按 macro-F1 排序
       ↓
train.py          →  解冻最后 2 个 block + head，DDP 微调
       ↓
evaluate.py       →  逐类指标 + 混淆矩阵
train.py          →  单图 / 目录批量推理
```

### 继承兄弟项目的三个关键选择

数据形态与兄弟项目 `image_classifier`（同源 ICT 数据，二分类）一致，本数据上已重新验证：

1. **letterbox（等比缩放 + 白边补齐），而不是 resize + centercrop。**
   裁剪会切掉长图两端的文字。本库 31.4% 的图长宽比 ≥ 2。实测缩到 336 后最短边 min=16.7px、中位数 213px，**没有任何一张被压塌**，letterbox 安全。

2. **温和增强，不做水平翻转、不做大裁剪。**
   翻转会破坏文字方向，大裁剪会切掉内容。只做小幅尺变/平移/旋转 + 轻微颜色抖动。

3. **三条硬结论写进默认值：**

| 结论 | 本项目的对应默认值 |
|---|---|
| **过拟合而非欠拟合**（train 0.97 / val 0.93） | `weight_decay 1e-3`、`augment mild`、`unfreeze_blocks 2`、`label_smoothing 0.1` |
| **分辨率不是杠杆**（448 ≈ 336） | 用 backbone 原生分辨率，不做无谓放大 |
| **标签噪声才是天花板** | 选模型用 `macro-f1`，不用 loss |

> 注意「过拟合」对应的杠杆是 **更强的正则**，**不是更高的学习率**。调参时别走错方向。

---

## 环境准备

已验证组合：**python 3.11 + torch 2.13.0 + transformers 5.15.0 + CUDA**。

```bash
conda create -n multi_classifier python=3.11 -y
conda activate multi_classifier

# torch 按目标机器的 CUDA 版本装，见 https://pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> **`transformers >= 5.15` 是硬性要求**：dinov3 的模型类型（`dinov3_vit`）在更早版本里不存在，加载 `dinov3-vitl16-pretrain` 会直接报未知模型类型。

---

## 模型权重

代码只依赖**本地目录**，不依赖 repo id。最可靠的方式是从源机器直接拷贝：

```bash
# 源机器
rsync -avP /tcfs/fs_dev/zhaoaohui/models/{clip-vit-large-patch14-336,dinov3-vitl16-pretrain,dinov2_vitl14,dinov2-base,siglip,siglip2-base-p32-256-ve} \
  新机器:/path/to/models/
```

6 个候选 backbone 的本地目录**建议与下表完全一致**（`configs/bakeoff.yaml` 按这些名字找模型），`runtime.models_root` 指向它们的父目录，换机器只改这一处。

| 配置名 | 本地目录名 | 大小 | 分辨率 | 特征维度 | HF repo |
|---|---|---:|---:|---:|---|
| `clip_l336` | `clip-vit-large-patch14-336` | 3.2G | 336 | 768 | `openai/clip-vit-large-patch14-336` ✅ |
| `dinov3_l16` | `dinov3-vitl16-pretrain` | 1.2G | 224 | 1024 | `facebook/dinov3-vitl16-pretrain-lvd1689m` ✅ |
| `dinov2_l14` | `dinov2_vitl14` | 2.3G | 224 | 1024 | ⚠️ 未核实 |
| `dinov2_b14` | `dinov2-base` | 661M | 224 | 768 | `facebook/dinov2-base` ✅ |
| `siglip_so400m` | `siglip` | 3.3G | 384 | 1152 | ⚠️ 未核实 |
| `siglip2_b32` | `siglip2-base-p32-256-ve` | 181M | 256 | 768 | ⚠️ 未核实 |

> ⚠️ 标「未核实」的三个：本地 checkpoint 里没有留下 `_name_or_path` 元数据，无法确认原始 repo id。优先用 rsync 拷贝；确实要重新下载请先在 HuggingFace 上确认 id。
>
> **`dinov3` 是 gated model**：需要先在模型页接受 Meta 的数据协议并 `hf login`（或设 `HF_TOKEN`），否则下载会返回 401/403。

无法直连 HuggingFace 时走镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
cd /path/to/models && bash hfd.sh openai/clip-vit-large-patch14-336
```

> **只跑一个模型也可以**：不必把 6 个都下完。跳过大比拼、直接用 `configs/exp/exp001_clip_l336.yaml` 即可，只下它用的那一个。CLIP 那个是最省事的选择——已核实 id、非 gated、兄弟项目验证过。

然后确认 `configs/base.yaml` 里这两处指向你的实际路径：

```yaml
data:
  root: multi_data/data0910/rejudged      # 图片根目录（一层子目录 = 类别名）
runtime:
  models_root: /path/to/models            # 模型权重根目录
  hf_offline: true                        # 离线环境设 true，禁止一切网络请求
```

模型目录放好后可以先自检：

```bash
python - <<'PY'
import sys; sys.path.insert(0, "scripts")
from config import load_config, resolve_model_path
from model import BackboneWrapper

cfg = load_config(["configs/base.yaml", "configs/bakeoff.yaml"])
for c in cfg["candidates"]:
    path = resolve_model_path(cfg, c)
    m = BackboneWrapper(c["backbone"], path, unfreeze_blocks=0, resolution=c["resolution"])
    assert m.output_dim == c["feat_dim"], f"{c['name']}: 实测 {m.output_dim} != 声明 {c['feat_dim']}"
    print(f"  {c['name']:<16} D={m.output_dim:<5} OK")
PY
```

输出维度与 `bakeoff.yaml` 里声明的 `feat_dim` 必须一致，对不上会直接断言失败而不是静默跑下去。

---

## 快速开始

如果你已经配好环境和权重，最快是这样一键跑：

```bash
# 改 configs/launch.yaml 里的 gpu_ids，然后用 launch 脚本串起全流程
python scripts/launch_pipeline.py --config configs/launch.yaml
```

`configs/launch.yaml` 里可以调：用几张卡、抽哪些 backbone、训练用哪个实验配置、跑哪些阶段。详见[多 GPU 运行](#多-gpu-运行)。

如果想单步手动跑，见下一节。

---

## 单步详解

### 第 1 步：划分数据

```bash
python scripts/prepare_data.py --config configs/base.yaml
```

做三件事：dHash 近重复分组 → 按类分层分配 train/val（**组整体分配**）→ 物化 `data/{train,val}/`（默认符号链接，不占空间）。

**预期输出**（关键数字对不上就说明有地方不对）：

```
图片 11529 张 → 11189 组（... 张与他图构成近重复）
[warn] 有 87 个组包含**不同标签**的近重复图片 ...      ← 这是标签噪声，符合预期
类别数 15（其中 5 类标记 unreliable、10 类进入主口径 macro-F1）
近重复组 11189 个，跨 split 泄漏 0 组（已断言）
「不相关」已按配置排除，未出现在划分中：True
```

**自检点**：`跨 split 泄漏 0 组`（脚本内是断言，泄漏会直接报错）；`不相关` 不出现；类别数 15。

### 第 2 步：抽冻结特征

```bash
python scripts/extract_features.py --config configs/base.yaml --config configs/bakeoff.yaml
```

对全部 11,529 张图各抽一份特征，存 `features/{name}.npy`（float16，约 24MB，可忽略）。
**backbone 冻结时特征与训练无关，抽一次可以反复用**。

中断了直接重跑，会自动接着抽剩下的：

```
[skip] clip_l336：缓存已完整（11529 张，D=768）
```

> 显存不够就调小 `features.batch_size`；单卡跑 6 个模型约 1–2 小时。只想跑一个：`--only clip_l336`。

### 第 3 步：backbone 大比拼

```bash
python scripts/linear_probe.py --config configs/base.yaml --config configs/bakeoff.yaml
```

在缓存特征上同一次评估里跑三种探针：**LogisticRegression**、**小 MLP**、**kNN**。输出 `outputs/bakeoff/bakeoff_summary.{csv,md}`，**按「10 个可靠类 macro-F1」排序**。

**怎么读**：
- 所有候选都该**显著高于随机水平 1/15 ≈ 0.067**。若都接近随机，问题在标签或划分，不在模型。
- 排名靠前的 backbone 才值得投入微调预算。
- 大模型不一定赢。本任务判别信号是文字与排版，小模型有时反而更好。

选中之后，把它写进实验 YAML（只写与 `base.yaml` 不同的部分）：

```yaml
# configs/exp/exp002_dinov3.yaml
model:
  backbone: dinov3
  model_name_or_path: dinov3-vitl16-pretrain
  resolution: 224
train:
  checkpoint_dir: checkpoints/exp002_dinov3
```

### 第 4 步：微调

```bash
# 单卡
python scripts/train.py --config configs/base.yaml --config configs/exp/exp001_clip_l336.yaml

# 单机多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29517 \
  scripts/train.py --config configs/base.yaml --config configs/exp/exp001_clip_l336.yaml
```

详见[多 GPU 运行](#多-gpu-运行)。

产物在 `checkpoints/exp001_clip_l336/`：`best.pt`、`last.pt`、`history.json`、`resolved_config.yaml`。`resolved_config.yaml` 记录了深合并后的完整配置 + git commit + python 版本，**每个实验目录自解释**。

### 第 5 步：评估

```bash
python scripts/evaluate.py --config configs/base.yaml \
  --override eval.checkpoint=checkpoints/exp001_clip_l336/best.pt
```

产出到 `outputs/eval/`：`report.txt`（逐类 P/R/F1）、`metrics.json`、`confusion_matrix.png`、`top_confusions.md`、`predictions.csv`。

加 `--split train` 可看训练集表现，判断过拟合。

### 第 6 步：推理

```bash
# 单张
python scripts/predict.py --config configs/base.yaml --input 某张图.jpg

# 目录批量，导出 CSV（含全部 15 类概率）
python scripts/predict.py --config configs/base.yaml --input 某个目录/ \
  --out preds.csv --show_all_probs
```

批量模式会打印**预测分布**——如果某类占了绝大多数，通常意味着模型塌缩或输入分布不对。

---

## 多 GPU 运行

### 推荐方式：launch 流水线脚本

项目提供 `scripts/launch_pipeline.py`，把「抽特征 → 比 backbone → 微调 → 评估」串起来，参数全放 `configs/launch.yaml`。

```bash
python scripts/launch_pipeline.py --config configs/launch.yaml
```

`configs/launch.yaml` 关键参数：

```yaml
# 用几张卡、用哪几张
gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]

# torchrun 通信端口，共享机器上务必改成不冲突的
master_port: 29517

# 跑哪些阶段，不需要就写 false
stages:
  prepare_data: true
  extract_features: true
  linear_probe: true
  train: true
  evaluate: true

# 抽哪些 backbone 的特征
extract_features:
  candidates:
    - clip_l336
    - dinov3_l16
    - dinov2_l14
    - dinov2_b14
    - siglip_so400m
    - siglip2_b32

# 训练用哪个实验配置
train:
  configs:
    - configs/base.yaml
    - configs/exp/exp001_clip_l336.yaml
  # 临时覆盖超参，不用改实验 YAML
  overrides:
    train.batch_size: 128
```

**各阶段占卡规则**：
- `prepare_data` / `linear_probe` / `evaluate`：单进程，不占多卡。
- `extract_features`：候选按 `gpu_ids` 循环分配，每张卡同时跑一个候选。8 卡 + 6 候选 → 6 张卡并行；4 卡 + 6 候选 → 先跑 4 个，再跑剩下 2 个。
- `train`：`torchrun --nproc_per_node=N` 启动 DDP，**占用 `gpu_ids` 全部卡**。

后台跑：

```bash
nohup python scripts/launch_pipeline.py --config configs/launch.yaml > launch.log 2>&1 &
tail -f launch.log
```

日志写在 `outputs/launch_logs/`，每个阶段一个文件。

### 手动方式：只跑训练的多卡

如果不需要 launch 脚本，手动跑 `train.py` 的多卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 --master_port=29517 \
  scripts/train.py \
  --config configs/base.yaml \
  --config configs/exp/exp001_clip_l336.yaml
```

**要点**：

| 事项 | 说明 |
|---|---|
| 选卡 | `CUDA_VISIBLE_DEVICES` 指定；`--nproc_per_node` **必须等于**卡数 |
| master_port | 共享机器**必须显式指定**，避免 29500 跟别人撞 |
| 后台 | `nohup ... > train.log 2>&1 &` |
| 确认生效 | 日志应有 `多卡：8 张卡，每卡 batch=...`；另开终端 `nvidia-smi` 确认 N 张卡在跑 |
| 每卡 batch | `train.batch_size` 是**全局**值，自动 ÷ 卡数。OOM 时调小它 |

> ⚠️ **不支持断点续训，中断 = 从 epoch 1 重来。** 长任务务必用 `nohup` / `tmux` 挂后台。

---

## 指标怎么读

**所有评估都报两个 macro-F1**：

| 指标 | 含义 | 什么时候看 |
|---|---|---|
| `macro-F1 (15 类全量)` | 包含 5 个 unreliable 类 | 与外部口径对齐时 |
| **`macro-F1 (仅可靠类)`** | 只统计 10 个样本充足的类 | **判断模型好坏就看这个**；`select_by` 也用它 |

原因：`质量类` 在 val 上只有 **2 张**，其 F1 只能取 0、0.5、1.0 三个值之一，单张翻转就能让 15 类 macro-F1 抖动 ±0.03 以上。

**看混淆矩阵时优先看行归一化那张**：不均衡数据下原始计数会被大类淹没。`top_confusions.md` 也按「占真实类样本的比例」排序，避免 4626 张的大类霸榜。

---

## 目录结构

```
multi_classifier/
├── README.md
├── requirements.txt
├── configs/
│   ├── classes.yaml          # 15 类清单、unreliable 标记
│   ├── base.yaml             # 全局默认
│   ├── bakeoff.yaml          # 6 个 backbone 候选
│   ├── launch.yaml           # 流水线启动配置
│   └── exp/                  # 每次实验加一份，只写与 base 的差异
├── scripts/
│   ├── config.py             # YAML 加载/深合并/校验
│   ├── launch_pipeline.py    # 多 GPU 一键流水线
│   ├── prepare_data.py       # 划分 + dHash 近重复防泄漏
│   ├── dataset.py            # letterbox / 温和增强
│   ├── model.py              # BackboneWrapper + ClassifierHead
│   ├── extract_features.py   # 冻结特征缓存
│   ├── linear_probe.py       # backbone 大比拼
│   ├── train.py              # 微调
│   ├── evaluate.py           # 逐类指标 + 混淆矩阵
│   └── predict.py            # 推理
├── docs/
│   ├── experiment_plan.md    # 实验计划与决策记录
│   └── launch_pipeline.md    # launch 脚本使用说明
├── data/                     # 划分产物
├── features/                 # 特征缓存
├── checkpoints/              # 训练产物
└── outputs/                  # 评估与对比报告
```

---

## 已知问题与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `ValueError: You have to specify input_ids` | CLIP/SigLIP 的 `AutoModel` 加载出来是双塔模型 | 代码走 `.vision_model(...)`，且仅当存在 `visual_projection` 时才投影 |
| `mat1 and mat2 must have the same dtype` | 某些 checkpoint 磁盘上就是 fp16 权重 | 加载后统一 `.float()`，混合精度交给 AMP |
| 解冻层数设了却不生效 | 各模型编码器路径不统一 | 代码用探测而非硬编码 |
| 分辨率报「不能被 patch_size 整除」 | SigLIP so400m 原生 384 / patch 14 除不尽 | 支持插值的模型只告警不报错 |
| 指标莫名掉一截，且不报错 | 训练与推理预处理不一致 | `evaluate.py` / `predict.py` 强制从 checkpoint 读回预处理参数 |
| 验证指标高得离谱 | 近重复图片被劈到 train/val 两侧 | 划分阶段组整体分配，脚本内断言 0 泄漏 |
| 类名对不上 | 原始目录叫 `器件资料类`，复核输出写 `器件资料` | 类名统一在 `configs/classes.yaml`，目录名别名自动映射 |

另外：`configs/base.yaml` 里分辨率的**规范位置是 `model.resolution`**。旧写法 `preprocess.resolution` 仍兼容但会打警告。

---

## 下一步方向

按性价比排序：

1. **看 `top_confusions.md`，人工抽查错例**。标签噪声是天花板，改标注的收益通常高于换模型。已确认的易混方向：`电路与工程设计图 → 产品与结构图片` / `→ 文档与认证图片`、`电子元器件图片 → 文档与认证图片`。
2. **确认 `unreliable` 类要不要合并或删除**。`质量类` 只有 10 张，留着会把 15 类 macro-F1 拖得很难看。
3. 换胜出的 backbone 试 `unfreeze_blocks` 4 / 6，若 val 不涨说明过拟合，**回头加强正则而不是加 lr**。
4. `class_weight` 在 `sqrt` / `auto` / `none` 之间比较。
5. 分辨率消融（`extract_features.py --include-ablation`）。预期收益很小，但成本也低。

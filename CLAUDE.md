# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ICT 图片 15 分类器。基于预训练视觉编码器 + 轻量分类头，把 ICT 领域图片分到 15 个细类。

关键范围约束：

- **不做相关性门控、不做拒识**。`不相关` 从数据划分开始就排除（`configs/classes.yaml` 的 `exclude`），当作不存在。外部环节负责过滤无关图片。
- **15 类单标签分类**，输出最可能类别 + 置信度 + top-3。
- 评估报 **两个 macro-F1**：15 类全量 vs. 仅 10 个可靠类。判断模型好坏看后者（`macro-F1 (仅可靠类)`）。
- 5 个 unreliable 类（`包装运输与仓储图片`、`器件资料`、`安装部署与运维图片`、`故障与维修图片`、`质量类`）照常训练，但不参与主口径。

项目记忆规则：

- 实验参数走 YAML 而非 CLI，保证可记录、可复现。
- 「不需要」= 就地移除 / 改 `false`，不是加门控或保留死代码。
- 脚本和 README 需要自包含，训练在另一台服务器上跑。

## Environment setup

已验证组合：python 3.11 + torch 2.13.0 + transformers 5.15.0 + CUDA。

```bash
conda create -n multi_classifier python=3.11 -y
conda activate multi_classifier
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`transformers>=5.15` 是硬性要求，否则 dinov3 会报未知模型类型。

## Model weights

代码只依赖**本地目录**，不依赖 HuggingFace repo id。把权重放到 `runtime.models_root` 下（默认 `/mnt/fs_dev/zhaoaohui/models`）。`configs/bakeoff.yaml` 里的 `model_name_or_path` 是相对于该根目录的目录名。

建议目录结构（与 `bakeoff.yaml` 候选一一对应）：

```
models_root/
├── clip-vit-large-patch14-336
├── dinov3-vitl16-pretrain
├── dinov2_vitl14
├── dinov2-base
├── siglip
└── siglip2-base-p32-256-ve
```

换机器时通常只改 `configs/base.yaml` 里的 `runtime.models_root`。

离线环境设 `runtime.hf_offline: true`（默认已开），禁止 HF 网络请求。

权重放好后可以自检：

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

## Common commands

### 一键跑完整流水线

```bash
python scripts/launch_pipeline.py --config configs/launch.yaml
```

`configs/launch.yaml` 控制用几张卡、抽哪些 backbone、训练用哪个实验配置、跑哪些阶段。后台跑：

```bash
nohup python scripts/launch_pipeline.py --config configs/launch.yaml > launch.log 2>&1 &
tail -f launch.log
```

日志写在 `outputs/launch_logs/`，每个阶段一个文件。

### 单步手动跑

```bash
# 1. 划分数据（CPU）
python scripts/prepare_data.py --config configs/base.yaml

# 2. 抽冻结特征
python scripts/extract_features.py --config configs/base.yaml --config configs/bakeoff.yaml
# 只跑一个候选
python scripts/extract_features.py --config configs/base.yaml --config configs/bakeoff.yaml --only clip_l336

# 3. backbone 大比拼
python scripts/linear_probe.py --config configs/base.yaml --config configs/bakeoff.yaml

# 4. 微调（单卡）
python scripts/train.py --config configs/base.yaml --config configs/exp/exp001_clip_l336.yaml

# 5. 评估
python scripts/evaluate.py --config configs/base.yaml \
  --override eval.checkpoint=checkpoints/exp001_clip_l336/best.pt

# 6. 推理
python scripts/predict.py --config configs/base.yaml --input path/to/image.jpg
python scripts/predict.py --config configs/base.yaml --input path/to/dir/ --out preds.csv --show_all_probs
```

### 多 GPU 微调

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29517 \
  scripts/train.py --config configs/base.yaml --config configs/exp/exp001_clip_l336.yaml
```

要点：

- `--nproc_per_node` 必须等于卡数。
- `master_port` 在共享机器上必须显式指定，避免 29500 冲突。
- `train.batch_size` 是**全局 batch**，脚本会自动按卡数均分到每卡。
- 不支持断点续训，中断后从 epoch 1 重来。

### 一次性覆盖（仅用于冒烟测试）

```bash
python scripts/train.py --config configs/base.yaml --config configs/exp/exp001_clip_l336.yaml \
  --override train.epochs=1 --override train.batch_size=32
```

正式实验的改动必须写进 YAML。

## Configuration system

配置由 `scripts/config.py` 统一加载：

- 多个 YAML 按命令行顺序**深合并**，后者覆盖前者。`base.yaml` 存全局默认，实验配置 `configs/exp/*.yaml` 只写差异。
- 支持 `--override key=value` 做一次性点号覆盖（如 `train.epochs=1`）。
- 相对路径统一按项目根目录解析，脚本可在任意 cwd 运行。
- 每个实验输出目录会写 `resolved_config.yaml`，包含完整合并配置 + git commit + python 版本，保证可复现。

关键配置键：

- `data.root`：图片根目录，一级子目录名即类别名。
- `data.class_config`：类别定义与 unreliable 标记，`configs/classes.yaml` 是全局唯一事实来源。
- `model.resolution`：分辨率的**规范位置**。`preprocess.resolution` 是历史遗留写法，仍兼容但会警告。
- `model.unfreeze_blocks`：解冻最后几个 block，0 表示完全冻结。
- `train.class_weight`：`auto` / `sqrt` / `none`；数据极不均衡，必须启用。
- `train.select_by`：`macro-f1`（默认，只看可靠类）/ `macro-f1-all` / `loss`。
- `runtime.models_root`：本地模型权重根目录。

## Code architecture

### Pipeline stages

```
prepare_data.py   →  划分数据 + dHash 近重复防泄漏
       ↓
extract_features.py  →  6 个 backbone 抽冻结特征缓存
       ↓
linear_probe.py   →  backbone 大比拼，按 macro-F1 排序
       ↓
train.py          →  解冻最后若干 block + head，DDP 微调
       ↓
evaluate.py       →  逐类指标 + 混淆矩阵
predict.py        →  单图 / 目录批量推理
```

`scripts/launch_pipeline.py` 串起上述阶段；每个阶段的参数都通过 YAML 控制，不硬编码在脚本里。

### Key modules

- `scripts/config.py`：YAML 深合并、路径解析、校验、`resolved_config.yaml` 落盘。
- `scripts/model.py`：`BackboneWrapper`（统一不同 backbone 的前向 / 解冻 / 分辨率处理）+ `ClassifierHead`。
- `scripts/dataset.py`：letterbox 缩放、温和增强、类别别名处理。
- `scripts/prepare_data.py`：dHash 近重复分组 + 分层划分 + 物化 `data/{train,val}/`。
- `scripts/extract_features.py`：冻结 backbone 抽特征，缓存到 `features/{name}.npy`，支持断点续跑。
- `scripts/linear_probe.py`：在缓存特征上同一次跑 LogisticRegression / MLP / kNN，输出 `outputs/bakeoff/`。
- `scripts/train.py`：DDP 微调，保存 `best.pt` / `last.pt` / `history.json`。
- `scripts/evaluate.py`：按 10 可靠类 / 15 类全量分别计算 macro-F1，生成混淆矩阵和 `top_confusions.md`。
- `scripts/predict.py`：推理，支持单图和目录批量。

### Data preprocessing

- **letterbox（等比缩放 + 白边补齐）**，不是 resize+centercrop。裁剪会切掉长图两端的文字。
- **温和增强**：只做小幅尺变/平移/旋转 + 轻微颜色抖动；不做水平翻转、不做大裁剪，避免破坏文字方向和内容。
- 推理和评估强制从 checkpoint 读回预处理参数，避免 train/test 预处理不一致。

### Training design

- 类别权重：数据极度不均衡（`电路与工程设计图 : 质量类 ≈ 463:1`），`train.class_weight` 默认 `sqrt`。
- 选模型用 `macro-f1`（仅可靠类），绝不用 accuracy 或 loss。
- 正则优先：默认 `weight_decay=1e-3`、`dropout=0.3`、`label_smoothing=0.1`、`unfreeze_blocks=2`。
- 分辨率默认用 backbone 原生分辨率，不盲目放大。

## Important constraints

- 实验参数必须进 YAML，CLI `--override` 只供临时冒烟。
- `不相关` / `硬件ICT其他` 必须在 `configs/classes.yaml` 的 `exclude` 中排除，不要加门控逻辑。
- 不可靠类指标不参与主口径，但训练时保留。
- 修改 `configs/classes.yaml` 的 `index` 顺序会改变模型输出下标，旧 checkpoint 会整体错位。

# 多 GPU 流水线启动说明

`scripts/launch_pipeline.py` 把全流程串起来，参数全部放在 `configs/launch.yaml` 里，
适合在 4/8 张 5090 上一键跑完。

## 快速开始

```bash
python scripts/launch_pipeline.py --config configs/launch.yaml
```

## 能调哪些参数

打开 `configs/launch.yaml`：

| 参数 | 含义 | 示例 |
|---|---|---|
| `gpu_ids` | 用哪几张卡 | `[0,1,2,3]` 或 `[0,1,2,3,4,5,6,7]` |
| `master_port` | torchrun 通信端口 | `29517` |
| `stages.*` | 是否执行该阶段 | `true` / `false |
| `extract_features.candidates` | 要抽特征的 backbone | 6 个候选任选 |
| `train.configs` | 训练用的 YAML | 通常 `base.yaml + exp/xxx.yaml` |
| `train.overrides` | 临时覆盖训练参数 | `train.batch_size: 128` |

## 各阶段怎么占卡

- **prepare_data / linear_probe / evaluate**：单进程，不占 GPU 或只用一张卡默认识别。
- **extract_features**：按 `gpu_ids` 循环分配候选，**每张卡同时跑一个候选**。
  - 8 卡 + 6 候选 → 6 张卡并行，2 张空闲。
  - 4 卡 + 6 候选 → 先并行 4 个，剩下 2 个等前面释放后继续。
- **train**：用 `torchrun --nproc_per_node=N` 启动 DDP，**占用 `gpu_ids` 全部卡**。

## 常见改法

### 只用 4 张卡

```yaml
gpu_ids: [0, 1, 2, 3]
```

### 只抽 3 个候选做快速验证

```yaml
extract_features:
  candidates:
    - clip_l336
    - dinov3_l16
    - dinov2_b14
```

### 训练时覆盖 batch size

```yaml
train:
  overrides:
    train.batch_size: 128
```

### 跳过某个阶段

```yaml
stages:
  prepare_data: false   # 数据已经划分好了
  extract_features: true
  linear_probe: true
  train: true
  evaluate: true
```

## 日志

每个阶段单独写 `.log` 到 `outputs/launch_logs/`：

```
outputs/launch_logs/
├── 00_prepare_data.log
├── 01_extract_features_clip_l336_gpu0.log
├── 01_extract_features_dinov3_l16_gpu1.log
├── 02_linear_probe.log
├── 03_train.log
└── 04_evaluate.log
```

## 后台运行

```bash
nohup python scripts/launch_pipeline.py --config configs/launch.yaml > launch.log 2>&1 &
tail -f launch.log
```

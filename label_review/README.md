# Label Review 目录说明

本目录存放多分类图片数据集的标签审核结果，按分类拆分为 `_errors.jsonl` 和 `_correct.jsonl` 成对文件。

---

## 目录结构

```
label_review/
├── README.md                           # 本文件
├── append_audit.py                     # 审核结果追加脚本
├── index.html                          # 前端可视化页面（浏览器打开即可查看）
├── all_errors.jsonl                    # 全部分类的 errors 聚合
├── all_correct.jsonl                   # 全部分类的 correct 聚合
├── {分类名}_errors.jsonl               # 该分类下标签错误的样本
├── {分类名}_correct.jsonl              # 该分类下标签正确的样本
└── _backup_20260913/                   # 历史备份
    └── *.orig
```

---

## JSONL 数据格式

每行一个 JSON 对象：

```json
{
  "image_path": "multi_data/data0910/raw/PCB与电路板图片/xxx.jpg",
  "current_label": "PCB与电路板图片",
  "reason": "审核理由说明...",
  "suggested_label": "器件资料类"
}
```

| 字段 | 说明 |
|------|------|
| `image_path` | 图片相对路径（相对于项目根目录） |
| `current_label` | 当前数据集分配的标签 |
| `reason` | 审核给出的判断理由 |
| `suggested_label` | 审核建议的正确标签 |

---

## Errors vs Correct 的判定规则

| 文件后缀 | 判定条件 | 含义 |
|----------|----------|------|
| `_errors.jsonl` | `suggested_label ≠ current_label` | 当前标签**错误**，需要修正 |
| `_correct.jsonl` | `suggested_label == current_label` | 当前标签**正确**，无需改动 |

> 特别地，`suggested_label` 为 `"不相关"` 表示该图片与 ICT 领域无关，建议从数据集中移除。

---

## 现有分类（13 个）

| 分类名 | _errors | _correct |
|--------|---------|----------|
| PCB与电路板图片 | ✓ | ✓ |
| 产品与结构图片 | ✓ | ✓ |
| 仿真分析图片 | ✓ | ✓ |
| 信号与波形图片 | ✓ | ✓ |
| 制造工艺图片 | ✓ | ✓ |
| 包装运输与仓储图片 | ✓ | ✓ |
| 器件资料类 | ✓ | ✓ |
| 安装部署与运维图片 | ✓ | ✓ |
| 工程实验图片 | ✓ | ✓ |
| 故障与维修图片 | ✓ | ✓ |
| 文档与认证图片 | ✓ | ✓ |
| 测试与检测图片 | ✓ | ✓ |
| 电子元器件图片 | ✓ | ✓ |
| 电路与工程设计图 | ✓ | ✓ |
| 质量类 | ✓ | ✓（均为空） |

---

## 工具脚本

### append_audit.py

将新的审核结果追加到对应分类的 JSONL 文件中，自动去重。

**用法：**

```bash
# 从 stdin 读取 JSON array，按 category 拆分并追加
python label_review/append_audit.py "PCB与电路板图片" < audit_results.json
```

**行为：**
- 将 `suggested_label ≠ category` 的样本写入 `{category}_errors.jsonl`
- 将 `suggested_label == category` 的样本写入 `{category}_correct.jsonl`
- 自动跳过已存在的 `image_path`（去重）
- 输出新增数量和重复数量

---

## 可视化查看

### 方式一：浏览器直接打开（需启动本地 HTTP 服务器）

在 `label_review/` 目录下启动简易 HTTP 服务器：

```bash
cd label_review
python3 -m http.server 8080 --bind 0.0.0.0
```

浏览器访问：
```
http://<服务器IP>:8080/index.html
```

### 方式二：从项目根目录启动

```bash
python3 -m http.server 8080 --bind 0.0.0.0
```

浏览器访问：
```
http://<服务器IP>:8080/label_review/index.html
```

### 可视化页面功能

- **自动发现**：自动扫描并列出目录下所有 `.jsonl` 文件
- **切换查看**：下拉菜单选择不同分类的 `_errors` 或 `_correct` 文件
- **筛选**：按 Errors / Correct 类型筛选、按建议标签筛选、关键词搜索
- **统计栏**：实时显示 Errors / Correct 数量及建议标签分布 TOP8
- **图片预览**：网格展示图片，点击放大查看详细信息

---

## 备份说明

`_backup_20260913/` 目录下存放了部分分类的原始 `.orig` 备份文件，供需要时回滚对比。

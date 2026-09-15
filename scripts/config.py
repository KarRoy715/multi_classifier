"""统一的 YAML 配置加载：多文件深合并 + 点号覆盖 + 校验。

设计要点
--------
1. **分层深合并**：按命令行给出的顺序依次合并，后者覆盖前者。这样 `base.yaml`
   存全局默认，`configs/exp/*.yaml` 只写与默认不同的部分，实验记录天然简洁。
2. **点号覆盖**：`--override train.epochs=1` 只用于冒烟测试这类一次性改动，
   不作为主要接口——可复现的实验一律写进 YAML。
3. **可复现性**：`dump_resolved()` 把合并后的完整配置写进实验输出目录，
   让每个实验目录自解释，不依赖「当时的命令行是什么」。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# 项目根目录（本文件位于 scripts/ 下）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """支持 config["a.b.c"] 与 attrs 风格访问的嵌套 dict。

    注意：这里只覆盖 `__getitem__`，键仍然真实存在于嵌套 dict 中，
    因此 `to_dict()` / `json.dump` / `yaml.dump` 都能原样工作。
    """

    def __getitem__(self, key: str) -> Any:
        if "." in key:
            node: Any = self
            for part in key.split("."):
                if not isinstance(node, dict) or part not in node:
                    raise KeyError(key)
                node = node[part]
            return node
        return super().__getitem__(key)

    def get_path(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并：dict 逐键递归，其它类型（含 list）整体替换。"""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_scalar(text: str) -> Any:
    """把 CLI 覆盖值解析成合适的类型：先当 YAML 标量读，失败就当字符串。"""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _apply_override(cfg: dict, expr: str) -> None:
    """应用一条 `a.b.c=value` 覆盖。中间层级不存在时自动创建。"""
    if "=" not in expr:
        raise ValueError(f"--override 格式应为 key=value，收到：{expr!r}")
    key, _, raw = expr.partition("=")
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ValueError(f"--override 的键为空：{expr!r}")

    node = cfg
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        if not isinstance(node[part], dict):
            raise ValueError(
                f"--override 无法下钻：{'.'.join(parts[:-1])} 处的值不是字典"
            )
        node = node[part]
    node[parts[-1]] = _parse_scalar(raw.strip())


_BACKBONES = ("clip", "dinov3", "dinov2", "siglip", "resnet")


def validate(cfg: Config) -> None:
    """尽早暴露配置错误，避免在训练跑了半小时后才崩。

    这里只做**不需要加载模型**就能做的检查；patch_size 整除性等依赖模型配置的校验
    放在 model.py 里做（那里能读到真实的 patch_size）。
    """
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"配置错误：{message}")

    backbone = cfg.get_path("model.backbone")
    require(
        backbone in _BACKBONES,
        f"model.backbone={backbone!r} 不在支持列表 {_BACKBONES} 内",
    )

    require(
        cfg.get_path("preprocess.mode") in ("letterbox", "crop"),
        f"preprocess.mode 必须是 letterbox 或 crop，收到 {cfg.get_path('preprocess.mode')!r}",
    )
    # resolution 的规范位置是 model.resolution（它描述的是 backbone 吃什么尺寸）；
    # preprocess.resolution 是历史遗留写法，仍接受但会提示，见 resolve_resolution()。
    for key in ("model.resolution", "preprocess.resolution"):
        resolution = cfg.get_path(key)
        require(
            resolution is None or (isinstance(resolution, int) and resolution > 0),
            f"{key} 必须是正整数或 null（原生分辨率），收到 {resolution!r}",
        )
    require(
        cfg.get_path("preprocess.augment") in ("none", "mild"),
        f"preprocess.augment 必须是 none 或 mild，收到 {cfg.get_path('preprocess.augment')!r}",
    )

    require(
        cfg.get_path("train.class_weight") in ("auto", "sqrt", "none"),
        f"train.class_weight 必须是 auto / sqrt / none，"
        f"收到 {cfg.get_path('train.class_weight')!r}",
    )
    require(
        cfg.get_path("train.select_by") in ("loss", "macro-f1", "macro-f1-all"),
        f"train.select_by 必须是 loss / macro-f1 / macro-f1-all，"
        f"收到 {cfg.get_path('train.select_by')!r}",
    )
    require(
        cfg.get_path("train.amp") in ("bf16", "fp16", "none"),
        f"train.amp 必须是 bf16 / fp16 / none，收到 {cfg.get_path('train.amp')!r}",
    )

    ratio = cfg.get_path("data.train_ratio")
    require(
        isinstance(ratio, (int, float)) and 0.0 < float(ratio) < 1.0,
        f"data.train_ratio 必须在 (0,1) 内，收到 {ratio!r}",
    )
    require(
        cfg.get_path("data.materialize_mode") in ("symlink", "copy"),
        f"data.materialize_mode 必须是 symlink 或 copy，"
        f"收到 {cfg.get_path('data.materialize_mode')!r}",
    )
    require(
        isinstance(cfg.get_path("data.phash_max_dist"), int)
        and cfg.get_path("data.phash_max_dist") >= 0,
        "data.phash_max_dist 必须是非负整数（0 表示关闭近重复分组）",
    )

    unfreeze = cfg.get_path("model.unfreeze_blocks")
    require(
        isinstance(unfreeze, int) and unfreeze >= 0,
        f"model.unfreeze_blocks 必须是非负整数，收到 {unfreeze!r}",
    )


def load_config(
    paths: list[str | Path],
    overrides: list[str] | None = None,
    *,
    resolve_paths: bool = True,
) -> Config:
    """加载并深合并若干 YAML，套用点号覆盖，返回 Config。

    paths 中的相对路径按项目根目录解析，因此脚本可以从任意 cwd 运行。
    """
    if not paths:
        paths = [PROJECT_ROOT / "configs" / "base.yaml"]

    merged: dict = {}
    used: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            candidate = PROJECT_ROOT / path
            path = candidate if candidate.exists() else path
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在：{path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"配置文件顶层必须是映射（dict）：{path}")
        merged = _deep_merge(merged, data)
        used.append(str(path))

    for expr in overrides or []:
        _apply_override(merged, expr)

    cfg = Config(merged)
    cfg["_config_files"] = used
    if resolve_paths:
        _resolve_relative_paths(cfg)
    validate(cfg)
    return cfg


# 这些键的值是「相对于项目根目录」的路径，加载后统一转成绝对路径，
# 这样脚本在任意 cwd 下都能正确工作。
_PATH_KEYS = (
    "data.root",
    "data.class_config",
    "data.splits_out",
    "data.materialize_dir",
    "features.out_dir",
    "train.checkpoint_dir",
    "eval.out_dir",
    "predict.out",
    "runtime.models_root",
)


def _resolve_relative_paths(cfg: Config) -> None:
    for key in _PATH_KEYS:
        value = cfg.get_path(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        # 用路径本身回写：Config 是嵌套 dict，需要按层级赋值
        parts = key.split(".")
        node: Any = cfg
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = str(path)


def resolve_model_path(cfg: Config, candidate: dict | None = None) -> str:
    """把候选里的 model_name_or_path 解析成绝对路径。

    已存在（本地模型库）就直接用；否则原样返回，交给 HF 按 repo id 处理。
    """
    name = (candidate or {}).get("model_name_or_path") or cfg["model"]["model_name_or_path"]
    path = Path(name)
    if path.is_absolute():
        return str(path)
    root = Path(cfg["runtime"]["models_root"])
    joined = root / path
    if joined.exists():
        return str(joined)
    # 不是本地目录，可能是 HF repo id（如 openai/clip-vit-base-patch32）
    return str(path)


def resolve_resolution(cfg: Config, candidate: dict | None = None) -> int | None:
    """统一解析本次要用的输入分辨率，返回 None 表示「用 backbone 原生分辨率」。

    存在三处可以声明分辨率，必须定一个明确的优先级，否则会出现
    「在 exp 里设了 model.resolution 却因为脚本读 preprocess.resolution 而不生效」
    这种不报错、只掉点的错配。优先级从高到低：

      1. candidate['resolution']        —— bakeoff.yaml 的候选自带（抽特征用）
      2. model.resolution               —— **规范位置**，实验文件写这里
      3. preprocess.resolution          —— 历史遗留写法，兼容但提示
      4. None                           —— 交给 native_resolution() 推断
    """
    if candidate and candidate.get("resolution") is not None:
        return int(candidate["resolution"])

    model_res = cfg.get_path("model.resolution")
    if model_res is not None:
        return int(model_res)

    legacy = cfg.get_path("preprocess.resolution")
    if legacy is not None:
        print(
            "[warn] 检测到 preprocess.resolution，该位置已废弃，"
            "请改写到 model.resolution（本次仍按旧值生效）。"
        )
        return int(legacy)
    return None


def setup_hf_env(cfg: Config) -> None:
    """按配置设置 HF 离线开关。离线时禁止任何网络请求，避免卡住等超时。"""
    offline = bool(cfg.get_path("runtime.hf_offline", True))
    os.environ.setdefault("HF_HUB_OFFLINE", "1" if offline else "0")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1" if offline else "0")
    # 关闭 tokenizers 的并发告警噪音
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_class_config(cfg: Config) -> dict:
    """读类别定义，校验 index 连续、与 num_classes 一致。"""
    path = Path(cfg["data"]["class_config"])
    with path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    classes = spec["classes"]
    indices = [c["index"] for c in classes]
    if indices != list(range(len(classes))):
        raise ValueError(
            f"classes.yaml 的 index 必须从 0 连续递增，实际为 {indices}"
        )
    if spec.get("num_classes") not in (None, len(classes)):
        raise ValueError(
            f"classes.yaml 的 num_classes={spec['num_classes']} "
            f"与 classes 数量 {len(classes)} 不一致"
        )
    spec["class_names"] = [c["name"] for c in classes]
    spec.setdefault("exclude", [])
    spec.setdefault("aliases", {})
    return spec


def save_resolved(cfg: Config, out_dir: str | Path, filename: str = "resolved_config.yaml") -> Path:
    """把合并后的完整配置 + 运行环境写进实验输出目录，让实验自解释。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    payload["_provenance"] = {
        "config_files": cfg.get("_config_files", []),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    path = out_dir / filename
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return path


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """给脚本挂上统一的 --config / --override 参数。"""
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="YAML",
        help="配置文件，可重复；按顺序深合并，后者覆盖前者。默认 configs/base.yaml",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="一次性点号覆盖（如 train.epochs=1），供冒烟测试用；正式实验请写进 YAML",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.override)


def get_override(overrides: dict[str, Any] | None, key: str) -> Any | None:
    """从 launch.yaml 的 flat overrides 字典里按字面量 dot-key 取值。

    Config.__getitem__ 会按 '.' 拆分嵌套路径，因此像 `train.checkpoint_dir`
    这种作为平面字典键存在的覆盖值不能直接用 cfg.get_path 读取。
    """
    if not overrides:
        return None
    return overrides.get(key)


def describe(cfg: Config) -> str:
    """一行摘要，脚本启动时打印，方便在日志里认实验。"""
    m = cfg["model"]
    t = cfg["train"]
    files = ", ".join(Path(p).name for p in cfg.get("_config_files", []))
    return (
        f"backbone={m['backbone']} model={Path(str(m['model_name_or_path'])).name} "
        f"res={cfg.get_path('preprocess.resolution') or 'native'} "
        f"unfreeze={m['unfreeze_blocks']} epochs={t['epochs']} lr={t['lr']} "
        f"wd={t['weight_decay']} ls={t['label_smoothing']} "
        f"cw={t['class_weight']} select={t['select_by']} | configs: {files}"
    )


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

"""模型定义：预训练视觉编码器（冻结）+ 轻量分类头。

支持的 backbone：clip / dinov3 / dinov2 / siglip / resnet

设计要点
--------
1. **统一取特征入口 `_extract_vision_features()`**。
   实测踩到的坑：`AutoModel.from_pretrained` 加载 CLIPModel / SiglipModel 得到的是
   **双塔模型**，直接 `model(pixel_values=x)` 会抛
   `ValueError: You have to specify input_ids`——必须走 `.vision_model(...)`，
   且只有存在 `visual_projection` 时才做投影（SigLIP 没有）。这条规则收在一处，
   避免每个 backbone 各写一遍出岔子。

2. **编码器层的定位用探测而非硬编码**。实测各模型的路径并不统一：
   clip/siglip → `vision_model.encoder.layers`，dinov2/dinov3 → `encoder.layer`
   （单数！），SiglipVisionModel → `encoder.layers`。硬编码很容易静默解冻错层。

3. **只给支持 `interpolate_pos_encoding` 的模型传该参数**。实测 clip/siglip 支持，
   dinov2/dinov3 不支持（DINOv3 用 RoPE，本来就不需要插值）。
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from transformers import AutoModel, CLIPModel  # noqa: F401  (CLIPModel 仅用于类型说明)


class ClassifierHead(nn.Module):
    """两层 MLP 分类头。线性探针阶段也复用这个结构，让探针结果与微调起点可比。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 15,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _extract_vision_features(
    model: nn.Module, pixel_values: torch.Tensor
) -> torch.Tensor:
    """从任意已加载的视觉模型里取出图像特征 (B, D)。

    双塔模型（CLIPModel / SiglipModel）：走 `.vision_model`，有 `visual_projection` 才投影。
    单塔模型（Dinov2Model / DINOv3ViTModel / SiglipVisionModel）：优先 pooler_output，
    没有就退回 CLS token（last_hidden_state[:, 0]）。
    """
    vision = getattr(model, "vision_model", None)
    if vision is not None:
        outputs = vision(pixel_values=pixel_values)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        projection = getattr(model, "visual_projection", None)
        # 只有 CLIP 系有 visual_projection；SigLIP 的 vision_model 输出已是最终嵌入
        return projection(pooled) if projection is not None else pooled

    kwargs: dict = {"pixel_values": pixel_values}
    # 只给真正支持该参数的模型传，否则会 TypeError
    if "interpolate_pos_encoding" in inspect.signature(model.forward).parameters:
        kwargs["interpolate_pos_encoding"] = True
    outputs = model(**kwargs)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled
    return outputs.last_hidden_state[:, 0]


# 各模型的编码器层可能挂在这些路径下，按顺序探测第一个命中的
_LAYER_PATHS = (
    "vision_model.encoder.layers",
    "encoder.layers",
    "encoder.layer",
    "model.encoder.layers",
    "model.layer",
    "model.layers",
    "layers",
)


def _find_encoder_layers(model: nn.Module) -> nn.ModuleList | None:
    """探测 transformer 编码器层列表。找不到返回 None（resnet 就没有）。"""
    for path in _LAYER_PATHS:
        node: nn.Module = model
        try:
            for part in path.split("."):
                node = getattr(node, part)
        except AttributeError:
            continue
        if isinstance(node, nn.ModuleList) and len(node) > 0:
            return node
    return None


def _patch_size(model: nn.Module) -> int | None:
    """读出 patch size，用于校验分辨率整除性。"""
    config = getattr(model, "config", None)
    if config is None:
        return None
    for src in (getattr(config, "vision_config", None), config):
        if src is None:
            continue
        patch = getattr(src, "patch_size", None)
        if isinstance(patch, int) and patch > 0:
            return patch
    return None


class BackboneWrapper(nn.Module):
    """加载并冻结预训练视觉编码器，可选解冻最后 N 个 block。"""

    def __init__(
        self,
        backbone_type: str,
        model_name: str,
        unfreeze_blocks: int = 0,
        resolution: int | None = None,
    ):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.model_name = str(model_name)
        self.unfreeze_blocks = unfreeze_blocks
        self.resolution = resolution
        self.is_hf_resnet = False

        if self.backbone_type == "resnet":
            self._build_resnet(self.model_name)
        elif self.backbone_type in ("clip", "dinov3", "dinov2", "siglip"):
            self._build_transformer(self.model_name, resolution)
        else:
            raise ValueError(
                f"未知 backbone：{backbone_type}。"
                "支持 clip / dinov3 / dinov2 / siglip / resnet"
            )

        # 全冻结时切到 eval 省掉 dropout/训练开销
        if not self.has_trainable_params:
            self.model.eval()

    # ---------- 构建 ----------

    def _build_transformer(self, model_name: str, resolution: int | None) -> None:
        self.model = AutoModel.from_pretrained(model_name)

        # 统一转 float32。
        # 实测坑：部分 checkpoint（如 siglip2-base-p32-256-ve）磁盘上就是 fp16 权重，
        # 加载后模型是 half，喂 float32 输入会报
        # 「mat1 and mat2 must have the same dtype, but got Half and Float」。
        # 统一在 float32 下跑，混合精度交给训练时的 AMP（bf16/fp16）处理。
        self.model = self.model.float()

        supports_interp = "interpolate_pos_encoding" in inspect.signature(
            (self.model.vision_model if hasattr(self.model, "vision_model") else self.model).forward
        ).parameters

        patch = _patch_size(self.model)
        if resolution is not None and patch:
            native = self._native_size()
            if resolution != native and resolution % patch != 0:
                message = (
                    f"分辨率 {resolution} 不是 {self.backbone_type} 的 patch_size={patch} 的整数倍"
                    f"（原生分辨率 {native}）。"
                )
                if supports_interp:
                    # SigLIP 就是这种情况：原生 384 / patch 14 本身就除不尽，
                    # 位置编码靠插值适配，因此只告警不报错。
                    print(f"[warn] {message}该模型支持位置编码插值，将按插值方式继续。")
                else:
                    raise ValueError(
                        f"{message}该模型不支持位置编码插值，请改成 {patch} 的倍数。"
                    )

        # CLIP 的位置编码严格对应一个固定网格，换分辨率必须插值扩展
        if self.backbone_type == "clip" and resolution is not None:
            native = self._clip_native_size()
            if resolution != native:
                self._interpolate_clip_pos_emb(native, patch, resolution)

        for param in self.model.parameters():
            param.requires_grad = False

        # 用一次 dummy forward 实测输出维度，而不是猜 hidden_size——
        # 双塔模型经过 visual_projection 后维度会和 hidden_size 不同（如 CLIP-L: 1024 → 768）
        self._output_dim = self._probe_output_dim(resolution)

        if self.unfreeze_blocks > 0:
            layers = _find_encoder_layers(self.model)
            if layers is None:
                raise AttributeError(
                    f"在 {self.backbone_type}（{Path(model_name).name}）里找不到编码器层列表，"
                    f"无法解冻最后 {self.unfreeze_blocks} 层。"
                    "可用 --override model.unfreeze_blocks=0 改为完全冻结。"
                )
            if self.unfreeze_blocks > len(layers):
                raise ValueError(
                    f"unfreeze_blocks={self.unfreeze_blocks} 超过该模型的层数 {len(layers)}"
                )
            for block in layers[-self.unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            # 双塔模型同时还应该解冻投影层，否则 CLIP 的视觉输出被冻住
            projection = getattr(self.model, "visual_projection", None)
            if projection is not None:
                for param in projection.parameters():
                    param.requires_grad = True

    def _native_size(self) -> int:
        """该模型配置里声明的原生分辨率（用于跳过不必要的分辨率告警）。"""
        config = getattr(self.model, "config", None)
        for src in (getattr(config, "vision_config", None), config):
            if src is None:
                continue
            size = getattr(src, "image_size", None)
            if isinstance(size, int):
                return size
        return 224

    def _probe_output_dim(self, resolution: int | None) -> int:
        """用 dummy 输入实测特征维度。"""
        size = resolution or self._guess_native_size()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, size, size, dtype=next(self.model.parameters()).dtype)
                return int(_extract_vision_features(self.model, dummy).shape[-1])
        finally:
            if was_training:
                self.model.train()

    def _guess_native_size(self) -> int:
        """兜底用的原生分辨率猜测（仅在没显式给 resolution 时用于探测维度）。"""
        from dataset import native_resolution

        return native_resolution(self.backbone_type, self.model_name)

    def _clip_native_size(self) -> int:
        config = getattr(self.model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        size = getattr(vision_config, "image_size", None)
        return int(size) if isinstance(size, int) else 224

    def _build_resnet(self, model_name: str) -> None:
        """torchvision 内置名，或本地 .pt/.pth/.bin / HF 格式目录。"""
        path = Path(model_name)
        if hasattr(torchvision.models, model_name):
            model_cls = getattr(torchvision.models, model_name)
            self.model = model_cls(weights="DEFAULT")
        elif path.is_dir():
            self.model = AutoModel.from_pretrained(str(path))
            self.is_hf_resnet = True
        elif path.exists():
            arch = _infer_resnet_arch(path.name)
            model_cls = getattr(torchvision.models, arch)
            self.model = model_cls(weights=None)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            self.model.load_state_dict(state)
        else:
            raise ValueError(
                f"无法识别的 ResNet：{model_name}。"
                "请用 resnet18/34/50/101/152，或本地 .pt/.pth/.bin 路径。"
            )

        for param in self.model.parameters():
            param.requires_grad = False

        if self.is_hf_resnet:
            self._output_dim = int(self.model.config.hidden_sizes[-1])
            if self.unfreeze_blocks > 0:
                for stage in self.model.encoder.stages[-self.unfreeze_blocks:]:
                    for param in stage.parameters():
                        param.requires_grad = True
        else:
            self._output_dim = int(self.model.fc.in_features)
            stages = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
            if self.unfreeze_blocks > 0:
                for stage in stages[-self.unfreeze_blocks:]:
                    for param in stage.parameters():
                        param.requires_grad = True

    def _interpolate_clip_pos_emb(self, native_img: int, patch_size: int | None, eff_res: int) -> None:
        """把 CLIP 的位置编码从原生分辨率双线性插值到 eff_res。

        保留 CLS token 的独立位置（索引 0 不动），只插值图像网格部分，
        并同步更新 image_size / num_patches / num_positions 等元数据。
        兼容「position_embeddings 是 Parameter」与「position_embedding 是 nn.Embedding」两种版本。
        """
        patch_size = patch_size or 14
        embeddings = self.model.vision_model.embeddings
        vision_config = self.model.config.vision_config

        if hasattr(embeddings, "position_embeddings"):
            pos_emb = embeddings.position_embeddings
            attr, is_embedding_module = "position_embeddings", False
        else:
            pos_emb = embeddings.position_embedding.weight.unsqueeze(0)
            attr, is_embedding_module = "position_embedding", True

        dim = pos_emb.shape[-1]
        native_grid = native_img // patch_size
        new_grid = eff_res // patch_size
        if new_grid == native_grid:
            return

        with torch.no_grad():
            cls_pos = pos_emb[:, :1]
            grid_pos = pos_emb[:, 1:].reshape(1, native_grid, native_grid, dim).permute(0, 3, 1, 2)
            grid_pos = F.interpolate(grid_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
            grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
            new_pos = torch.cat([cls_pos, grid_pos], dim=1)

        num_positions = 1 + new_grid * new_grid
        if is_embedding_module:
            new_embedding = nn.Embedding(num_positions, dim, device=new_pos.device, dtype=new_pos.dtype)
            new_embedding.weight = nn.Parameter(new_pos.squeeze(0), requires_grad=False)
            setattr(embeddings, attr, new_embedding)
        else:
            setattr(embeddings, attr, nn.Parameter(new_pos, requires_grad=False))

        # 新版 transformers 会显式校验这些元数据，不同步会直接报维度错误
        embeddings.image_size = eff_res
        embeddings.num_patches = new_grid * new_grid
        embeddings.num_positions = num_positions
        if "position_ids" in getattr(embeddings, "_buffers", {}):
            device = embeddings._buffers["position_ids"].device
            embeddings._buffers["position_ids"] = torch.arange(
                num_positions, dtype=torch.long, device=device
            ).unsqueeze(0)
        vision_config.image_size = eff_res

    # ---------- 属性与前进 ----------

    @property
    def has_trainable_params(self) -> bool:
        return any(p.requires_grad for p in self.model.parameters())

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == "resnet" and not self.is_hf_resnet:
            x = self.model.conv1(pixel_values)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        if self.backbone_type == "resnet" and self.is_hf_resnet:
            outputs = self.model(pixel_values=pixel_values, return_dict=True)
            return outputs.pooler_output.flatten(start_dim=1)
        return _extract_vision_features(self.model, pixel_values)


def _infer_resnet_arch(name: str) -> str:
    lowered = name.lower()
    for arch in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        if arch in lowered or arch.replace("resnet", "resnet-") in lowered:
            return arch
    raise ValueError(
        f"无法从 {name!r} 推断 ResNet 架构，路径里应包含 "
        "resnet18/34/50/101/152 之一。"
    )


def build_model(
    backbone: str,
    model_name: str,
    num_classes: int,
    unfreeze_blocks: int = 0,
    resolution: int | None = None,
    hidden_dim: int = 256,
    dropout: float = 0.3,
) -> tuple[BackboneWrapper, ClassifierHead]:
    wrapper = BackboneWrapper(
        backbone, model_name, unfreeze_blocks=unfreeze_blocks, resolution=resolution
    )
    head = ClassifierHead(
        input_dim=wrapper.output_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
    return wrapper, head


def load_backbone_from_checkpoint(ckpt: dict, models_root: str | None = None) -> BackboneWrapper:
    """按 checkpoint 里记录的 backbone / 模型路径 / 分辨率重建冻结编码器。

    evaluate.py 与 predict.py 都强制走这里，用 checkpoint 的值覆盖配置，
    避免「训练用 letterbox@336、推理却用 crop@224」这种静默掉点。
    """
    model_name = ckpt.get("model_name_or_path") or ckpt.get("model_name")
    if model_name and models_root:
        candidate = Path(models_root) / str(model_name)
        if candidate.exists():
            model_name = str(candidate)
    return BackboneWrapper(
        backbone_type=ckpt["backbone"],
        model_name=str(model_name),
        unfreeze_blocks=ckpt.get("unfreeze_blocks", 0),
        resolution=ckpt.get("resolution"),
    )

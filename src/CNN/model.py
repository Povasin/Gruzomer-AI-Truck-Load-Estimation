"""CNN-модель: предобученный backbone (timm) + регрессионная голова + две
вспомогательные (aux) классификационные головы под load_bin и cargo_type.

Aux-головы используются только во время обучения как регуляризатор
(multi-task learning) — на инференсе используется только load_pct.
"""

import timm
import torch
import torch.nn as nn

from manual.feature_schema import TOTAL_FEATURE_DIM


class TruckLoadNet(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        pretrained=True,
        num_load_bins=10,
        num_cargo_types=5,
        dropout=0.3,
    ):
        super().__init__()
        self._backbone_trainable = True
        # num_classes=0 + global_pool="avg" -> backbone возвращает уже
        # усреднённый вектор признаков (без своей classification head).
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features

        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(feature_dim, 1)
        self.load_bin_head = nn.Linear(feature_dim, num_load_bins)
        self.cargo_type_head = nn.Linear(feature_dim, num_cargo_types)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)

        # sigmoid * 100 держит регрессию в физически корректном диапазоне
        # [0, 100] без необходимости clip на инференсе.
        load_pct = torch.sigmoid(self.regression_head(features)).squeeze(-1) * 100.0
        load_bin_logits = self.load_bin_head(features)
        cargo_type_logits = self.cargo_type_head(features)

        return load_pct, load_bin_logits, cargo_type_logits

    def set_backbone_trainable(self, trainable):
        """Фаза 1: backbone заморожен, обучаются только головы.
        Фаза 2: вызывается с trainable=True, backbone разморожен целиком.
        """
        for param in self.backbone.parameters():
            param.requires_grad = trainable
        self._backbone_trainable = trainable
        self.backbone.train(self.training and trainable)

    def train(self, mode=True):
        super().train(mode)
        if not self._backbone_trainable:
            self.backbone.eval()
        return self


class HybridTruckLoadNet(TruckLoadNet):
    """CNN-вектор и числовые признаки объединяются перед регрессией."""

    numeric_dim = TOTAL_FEATURE_DIM

    def __init__(self, backbone_name="resnet18", pretrained=True,
                 numeric_mean=None, numeric_scale=None):
        super().__init__(backbone_name=backbone_name, pretrained=pretrained)
        mean = torch.zeros(self.numeric_dim) if numeric_mean is None else torch.as_tensor(numeric_mean, dtype=torch.float32)
        scale = torch.ones(self.numeric_dim) if numeric_scale is None else torch.as_tensor(numeric_scale, dtype=torch.float32)
        if mean.shape != (self.numeric_dim,) or scale.shape != (self.numeric_dim,):
            raise ValueError(f"Нормализация должна содержать {self.numeric_dim} значений")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Некорректные параметры нормализации")
        self.register_buffer("numeric_mean", mean.clone())
        self.register_buffer("numeric_scale", scale.clone())
        self.regression_head = nn.Sequential(
            nn.Linear(self.backbone.num_features + self.numeric_dim, 128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1),
        )

    def forward(self, images, numeric):
        if numeric.ndim != 2 or numeric.shape != (images.shape[0], self.numeric_dim):
            raise ValueError(f"Ожидается матрица признаков [batch, {self.numeric_dim}]")
        if not torch.isfinite(numeric).all():
            raise ValueError("Признаки содержат NaN/Inf")
        visual = self.dropout(self.backbone(images))
        normalized = (numeric - self.numeric_mean) / self.numeric_scale
        combined = torch.cat((visual, normalized), dim=1)
        load_pct = torch.sigmoid(self.regression_head(combined)).squeeze(-1) * 100.0
        return load_pct, self.load_bin_head(visual), self.cargo_type_head(visual)

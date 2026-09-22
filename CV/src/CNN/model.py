"""CNN-модель: предобученный backbone (timm) + регрессионная голова + две
вспомогательные (aux) классификационные головы под load_bin и cargo_type.

Aux-головы используются только во время обучения как регуляризатор
(multi-task learning) — на инференсе используется только load_pct.
"""

import timm
import torch
import torch.nn as nn

from manual.feature_schema import TOTAL_FEATURE_DIM


NUM_PERCENT_BINS = 21
PERCENT_BIN_CENTERS = torch.linspace(0, 100, NUM_PERCENT_BINS)


def get_soft_targets(y, num_bins=NUM_PERCENT_BINS, smoothing=0.01):
    """Преобразует target_pct [B] в вероятностное распределение по 21 бину (0, 5, ..., 100).
    Математическое ожидание распределения точно соответствует y.
    """
    device = y.device
    step = 100.0 / (num_bins - 1)
    y_clamped = torch.clamp(y, 0.0, 100.0)
    idx_low = torch.clamp((y_clamped / step).long(), 0, num_bins - 2)
    idx_high = idx_low + 1
    alpha = (y_clamped - idx_low.float() * step) / step

    targets = torch.zeros(len(y), num_bins, device=device)
    targets.scatter_add_(1, idx_low.unsqueeze(1), (1.0 - alpha).unsqueeze(1))
    targets.scatter_add_(1, idx_high.unsqueeze(1), alpha.unsqueeze(1))

    if smoothing > 0:
        targets = (1.0 - smoothing) * targets + (smoothing / num_bins)
    return targets


class TruckLoadNet(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        pretrained=True,
        num_load_bins=10,
        num_cargo_types=5,
        num_percent_bins=NUM_PERCENT_BINS,
        head_type="distributional",
        dropout=0.3,
    ):
        super().__init__()
        self._backbone_trainable = True
        self.head_type = head_type
        self.num_percent_bins = num_percent_bins
        self.register_buffer("bin_centers", torch.linspace(0, 100, num_percent_bins))

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
        self.dist_head = nn.Linear(feature_dim, num_percent_bins)
        self.load_bin_head = nn.Linear(feature_dim, num_load_bins)
        self.cargo_type_head = nn.Linear(feature_dim, num_cargo_types)

    def extract_features(self, x):
        """Извлекает глобальный визуальный вектор признаков из backbone."""
        return self.backbone(x)

    def forward(self, x, return_dist=False):
        features = self.backbone(x)
        features = self.dropout(features)

        dist_logits = self.dist_head(features)
        dist_probs = torch.softmax(dist_logits, dim=-1)
        expected_pct = torch.sum(dist_probs * self.bin_centers.to(dist_probs.device), dim=-1)
        scalar_pct = torch.sigmoid(self.regression_head(features)).squeeze(-1) * 100.0

        if self.head_type == "distributional":
            load_pct = expected_pct
        elif self.head_type == "blend":
            load_pct = 0.5 * expected_pct + 0.5 * scalar_pct
        else:
            load_pct = scalar_pct

        load_bin_logits = self.load_bin_head(features)
        cargo_type_logits = self.cargo_type_head(features)

        if return_dist:
            return load_pct, load_bin_logits, cargo_type_logits, dist_logits
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

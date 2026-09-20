"""Общий формат числовых признаков без импорта моделей сегментации."""

BASELINE_FEATURE_DIM = 235
FLOOR_FEATURE_DIM = 42
CEILING_FEATURE_DIM = 42
CROSS_FEATURE_DIM = 30
TOTAL_FEATURE_DIM = (
    BASELINE_FEATURE_DIM + FLOOR_FEATURE_DIM + CEILING_FEATURE_DIM + CROSS_FEATURE_DIM
)
FEATURE_VERSION = "truck-floor-ceiling-cross-349-masked-roi-v2"

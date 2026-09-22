from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CV_SRC = PROJECT_ROOT / "CV" / "src"
if str(CV_SRC) not in sys.path:
    sys.path.insert(0, str(CV_SRC))

from predictor import InvalidImageError, TruckLoadPredictor

DEFAULT_MODEL_PATH = PROJECT_ROOT / "CV" / "models" / "cnn_v1"
_predictor: TruckLoadPredictor | None = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _get_predictor() -> TruckLoadPredictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = TruckLoadPredictor(
                weights_path=os.getenv("CV_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
                device=os.getenv("CV_DEVICE", "auto"),
                use_tta=_env_bool("CV_USE_TTA", True),
                use_clahe_tta=_env_bool("CV_USE_CLAHE_TTA", True),
                calibrate=_env_bool("CV_CALIBRATE", True),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("CV predictor is unavailable") from exc
    return _predictor


def estimate_load(image_bytes: bytes) -> dict[str, Any]:
    """Estimate a cargo-bay image and return the complete service payload."""
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise InvalidImageError("Изображение должно быть непустым набором байтов")
    return _get_predictor().predict(image_bytes)

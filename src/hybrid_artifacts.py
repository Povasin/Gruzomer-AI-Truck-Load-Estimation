"""Версионированный NPZ гибрида: только массивы и JSON, без pickle."""

import json
from pathlib import Path

import numpy as np
import torch

from model import HybridTruckLoadNet

FORMAT_VERSION = 1


def save_checkpoint(path, model, metadata):
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Гибрид сохраняется в .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"state." + key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Модель содержит NaN/Inf")
    arrays.update(method=np.asarray("hybrid"), format_version=np.asarray(FORMAT_VERSION),
                  metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False)))
    # Режим x защищает существующие веса, в том числе результаты друга.
    with path.open("xb") as file:
        np.savez_compressed(file, **arrays)


def read_metadata(path):
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["method"]) != "hybrid" or int(saved["format_version"]) != FORMAT_VERSION:
            raise ValueError("Неподдерживаемый формат гибридной модели")
        return json.loads(str(saved["metadata"]))


def load_checkpoint(path):
    metadata = read_metadata(path)
    model = HybridTruckLoadNet(backbone_name=metadata["backbone"], pretrained=False)
    with np.load(path, allow_pickle=False) as saved:
        state = {key[6:]: torch.from_numpy(saved[key].copy()) for key in saved.files if key.startswith("state.")}
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Модель содержит NaN/Inf")
    model.load_state_dict(state, strict=True)
    if (model.numeric_scale <= 0).any():
        raise ValueError("Некорректная нормализация в модели")
    return model.eval(), metadata

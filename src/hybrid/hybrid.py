"""Единое обучение и прогноз CNN + числовые признаки на фиксированном split."""

import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from CNN.dataset import build_transforms, LOAD_BIN_TO_IDX, CARGO_TYPE_TO_IDX
from hybrid.hybrid_artifacts import load_checkpoint, read_metadata, save_checkpoint
from hybrid.hybrid_features import (
    FEATURE_DIM,
    build_contract, cached_samples, fit_normalization, resolve_saved_path,
    validate_features, verify_contract,
)
from CNN.model import HybridTruckLoadNet
from report import read_split_table, read_test_table, validate_fixed_split, make_split_diagnostics
from submission import write_submission


class HybridDataset(Dataset):
    """Детерминированный masked ROI и признаки в одном порядке строк."""

    def __init__(self, rows, sample_entries, img_size, numeric, targets=False):
        if len(rows) != len(sample_entries):
            raise ValueError("Количество ROI не совпадает с количеством строк")
        self.rows = rows
        self.sample_entries = [Path(path) for path in sample_entries]
        self.transform = build_transforms(train=False, img_size=img_size)
        self.numeric = validate_features(numeric, len(rows))
        self.targets = targets
        if targets:
            for row in rows:
                if row["load_bin"] not in LOAD_BIN_TO_IDX or row["cargo_type"] not in CARGO_TYPE_TO_IDX:
                    raise ValueError(f"Неизвестная категория для {row['image_id']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with np.load(self.sample_entries[index], allow_pickle=False) as saved:
            roi = saved["roi"]
        if roi.dtype != np.uint8 or roi.ndim != 3 or roi.shape[2] != 3:
            raise ValueError(f"Некорректный cached ROI: {self.sample_entries[index]}")
        item = {
            "image": self.transform(image=roi)["image"],
            "image_id": self.rows[index]["image_id"],
            "numeric": torch.from_numpy(self.numeric[index].copy()),
        }
        if self.targets:
            row = self.rows[index]
            item.update(load_pct=torch.tensor(float(row["load_pct"]), dtype=torch.float32),
                        load_bin=torch.tensor(LOAD_BIN_TO_IDX[row["load_bin"]], dtype=torch.long),
                        cargo_type=torch.tensor(CARGO_TYPE_TO_IDX[row["cargo_type"]], dtype=torch.long))
        return item


def validate_args(args):
    for name, minimum in (("phase1_epochs", 0), ("phase2_epochs", 1), ("batch_size", 2),
                          ("num_workers", 0), ("patience", 1)):
        if getattr(args, name) < minimum:
            raise ValueError(f"{name} должно быть не меньше {minimum}")
    if args.img_size < 32 or args.img_size % 32:
        raise ValueError("img_size должно быть не меньше 32 и кратно 32")
    for name in ("head_lr", "backbone_lr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} должно быть конечным положительным числом")
    if not math.isfinite(args.aux_weight) or args.aux_weight < 0:
        raise ValueError("aux_weight должно быть конечным неотрицательным числом")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed должен быть в [0, 2**32)")


def output_paths(args):
    model = Path(args.model).resolve()
    if model.suffix.lower() != ".npz":
        raise ValueError("Для гибрида --model должен иметь расширение .npz")
    paths = {"model": model, "validation_model": model.with_name(model.stem + ".validation.npz"),
             "report": model.with_suffix(".json"), "errors": Path(args.errors_output).resolve()}
    if len(set(paths.values())) != len(paths):
        raise ValueError("Пути модели, отчёта и CSV ошибок должны различаться")
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Результат уже существует; укажите новое имя: {path}")
    return paths


def get_device(name, amp=False):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна; используйте --device cpu")
    if amp and name != "cuda":
        raise ValueError("--amp поддерживается только с CUDA")
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, args, train):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train,
                      num_workers=args.num_workers, drop_last=False,
                      generator=torch.Generator().manual_seed(args.seed))


def predict_batches(model, loader, device):
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in loader:
            prediction, _, _ = model(batch["image"].to(device), batch["numeric"].to(device))
            values.extend(prediction.float().cpu().numpy().tolist())
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
        raise ValueError("Некорректные прогнозы гибрида")
    return values


def fit_model(args, dataset, normalization, device, validation_dataset=None, epochs=None):
    """Тяжёлый этап: вызывается только при явном запуске команды train."""
    seed_everything(args.seed)
    model = HybridTruckLoadNet(backbone_name=args.backbone, pretrained=not args.no_pretrained,
                               numeric_mean=normalization[0], numeric_scale=normalization[1]).to(device)
    train_loader = make_loader(dataset, args, train=True)
    val_loader = make_loader(validation_dataset, args, train=False) if validation_dataset is not None else None
    total_epochs = epochs if epochs is not None else args.phase1_epochs + args.phase2_epochs
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    best_mae, best_epoch, best_state, best_prediction = float("inf"), 0, None, None
    stale = 0
    history = []
    optimizer = None
    for epoch in range(total_epochs):
        if epoch == 0 or epoch == args.phase1_epochs:
            unfrozen = epoch >= args.phase1_epochs
            model.set_backbone_trainable(unfrozen)
            heads = [parameter for name, parameter in model.named_parameters()
                     if not name.startswith("backbone.")]
            groups = [{"params": heads, "lr": args.head_lr}]
            if unfrozen:
                groups.append({"params": model.backbone.parameters(), "lr": args.backbone_lr})
            optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
            stale = 0
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            # На ResNet последний batch может содержать один объект с картой 1x1.
            # BatchNorm использует накопленные статистики только для такого batch.
            if len(batch["image"]) == 1:
                model.backbone.eval()
            else:
                model.backbone.train(model._backbone_trainable)
            with torch.autocast(device_type=device.type, enabled=args.amp):
                pred, bins, cargo = model(batch["image"].to(device), batch["numeric"].to(device))
                loss = nn.functional.l1_loss(pred, batch["load_pct"].to(device))
                loss = loss + args.aux_weight * (
                    nn.functional.cross_entropy(bins, batch["load_bin"].to(device))
                    + nn.functional.cross_entropy(cargo, batch["cargo_type"].to(device)))
            if not torch.isfinite(loss):
                raise ValueError("Обучение остановлено: loss содержит NaN/Inf")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(batch["image"])
        row = {"epoch": epoch + 1, "phase": 1 if epoch < args.phase1_epochs else 2,
               "train_loss": loss_sum / len(dataset)}
        if val_loader is not None:
            prediction = predict_batches(model, val_loader, device)
            actual = np.asarray([float(row["load_pct"]) for row in validation_dataset.rows])
            score = float(np.abs(actual - prediction).mean())
            row["val_mae"] = score
            if score < best_mae:
                best_mae, best_epoch = score, epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_prediction = prediction.copy()
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_loader is not None and epoch >= args.phase1_epochs and stale >= args.patience:
            break
    if val_loader is not None:
        if best_state is None:
            raise ValueError("Не удалось выбрать конечную модель по validation")
        model.load_state_dict(best_state)
    else:
        best_epoch = total_epochs
    return model.cpu().eval(), best_epoch, best_prediction, history


def train(args):
    from train import metrics, write_validation_errors

    validate_args(args)
    paths = output_paths(args)
    device = get_device(args.device, args.amp)
    train_rows = read_split_table(args.train_split)
    val_rows = read_split_table(args.validation_split)
    split_info = validate_fixed_split(train_rows, val_rows)
    for row in train_rows + val_rows:
        if row["load_bin"] not in LOAD_BIN_TO_IDX or row["cargo_type"] not in CARGO_TYPE_TO_IDX:
            raise ValueError(f"Неизвестная категория для {row['image_id']}")
        if not (Path(args.images) / f"{row['image_id']}.jpg").is_file():
            raise FileNotFoundError(f"Нет изображения {row['image_id']}")
    contract = build_contract(args.truck_weights, args.floor_weights)
    cache = args.feature_cache or paths["model"].parent / "hybrid_feature_cache"
    x_train, train_entries = cached_samples(args.images, train_rows, contract, cache)
    x_val, val_entries = cached_samples(args.images, val_rows, contract, cache)
    train_dataset = HybridDataset(train_rows, train_entries, args.img_size, x_train, targets=True)
    val_dataset = HybridDataset(val_rows, val_entries, args.img_size, x_val, targets=True)
    normalization = fit_normalization(x_train)
    print(f"Обучение CNN + {FEATURE_DIM} признаков; устройство: {device}", flush=True)
    model, best_epoch, prediction, history = fit_model(args, train_dataset, normalization, device,
                                                     validation_dataset=val_dataset)
    y_val = np.asarray([float(row["load_pct"]) for row in val_rows])
    if prediction.shape != y_val.shape or not np.isfinite(prediction).all():
        raise ValueError("Неполный или некорректный прогноз validation")
    validation_metrics = metrics(y_val, prediction)
    config_keys = ("backbone", "img_size", "seed", "batch_size", "phase1_epochs", "phase2_epochs",
                   "head_lr", "backbone_lr", "aux_weight", "patience", "amp", "no_pretrained")
    metadata = {key: getattr(args, key) for key in config_keys}
    metadata.update(feature_contract=contract, preprocessing="masked-truck-roi-letterbox-imagenet-v2",
                    best_epoch=best_epoch, load_bin_classes=list(LOAD_BIN_TO_IDX),
                    cargo_type_classes=list(CARGO_TYPE_TO_IDX))
    save_checkpoint(paths["validation_model"], model,
                    {**metadata, "role": "validation", "fit_count": len(train_rows)})
    write_validation_errors(paths["errors"], val_rows, y_val, prediction)
    del model
    all_rows = train_rows + val_rows
    x_all = np.concatenate((x_train, x_val), axis=0)
    all_dataset = HybridDataset(
        all_rows,
        train_entries + val_entries,
        args.img_size,
        x_all,
        targets=True,
    )
    print(f"Финальное обучение на train + validation: {best_epoch} эпох", flush=True)
    final_model, _, _, final_history = fit_model(args, all_dataset, fit_normalization(x_all), device,
                                                epochs=best_epoch)
    save_checkpoint(paths["model"], final_model, {**metadata, "role": "final", "fit_count": len(all_rows)})
    report = {"selected_method": "hybrid", "features": FEATURE_DIM, "hybrid": validation_metrics,
              "split_type": "fixed_precomputed", **split_info,
              "train_split": str(args.train_split), "validation_split": str(args.validation_split),
              "split_diagnostics": make_split_diagnostics(train_rows, val_rows),
              "configuration": metadata, "history": history, "final_history": final_history,
              "artifacts": {key: str(value) for key, value in paths.items()},
              "note": "MAE относится к validation checkpoint, обученному только на train. Финальная модель обучена заново на train + validation и здесь не оценивается."}
    with paths["report"].open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(validation_metrics, ensure_ascii=False), flush=True)
    print(f"Модель: {paths['model']}\nОтчёт: {paths['report']}", flush=True)


def predict(args):
    metadata = read_metadata(args.model)
    if metadata.get("preprocessing") != "masked-truck-roi-letterbox-imagenet-v2":
        raise ValueError("Неизвестная обработка изображения в модели")
    saved_contract = metadata["feature_contract"]
    truck = args.truck_weights or resolve_saved_path(saved_contract["truck_weights"])
    floor = args.floor_weights or resolve_saved_path(saved_contract["floor_weights"])
    contract = build_contract(truck, floor)
    verify_contract(saved_contract, contract)
    rows = read_test_table(args.test_csv)
    cache = args.feature_cache or Path(args.model).resolve().parent / "hybrid_feature_cache"
    numeric, entries = cached_samples(args.images, rows, contract, cache)
    dataset = HybridDataset(rows, entries, metadata["img_size"], numeric)
    device = get_device(args.device)
    model, _ = load_checkpoint(args.model)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    prediction = predict_batches(model.to(device), loader, device)
    write_submission(rows, prediction, args.output)

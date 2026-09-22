"""4-Fold CNN (ResNet-18) + 349 Geometric Features Stacking Pipeline.

1-й уровень:
    4-фолдовый ансамбль свёрточных моделей TruckLoadNet (ResNet-18),
    обученных по GroupKFold (folds.csv) с multi-task loss.
    Генерирует Out-Of-Fold (OOF) прогнозы на train и усреднённый прогноз с TTA на test.

2-й уровень:
    Мета-модель (Ridge-регрессия), которая обучается объединять честный
    прогноз CNN и 349 геометрических признаков (кузов + пол + потолок).

3-й уровень:
    Пороговая калибровка крайних значений (<= 3.5% -> 0.0%, >= 94.5% -> 100.0%).
"""

import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from CNN.model import TruckLoadNet
from CNN.dataset import build_transforms, read_rgb_image
from manual.feature_schema import TOTAL_FEATURE_DIM
from hybrid.hybrid_features import build_contract, cached_features, resolve_saved_path
from report import read_test_table
from submission import write_submission


def get_device(device_arg="auto"):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_cnn_model(ckpt_path, device):
    """Загружает один чекпоинт TruckLoadNet."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "resnet18")
    state = ckpt.get("model_state_dict", {})
    head_type = ckpt.get("head_type", "distributional" if "dist_head.weight" in state else "scalar")
    img_size = ckpt.get("img_size", 320)
    
    model = TruckLoadNet(backbone_name=backbone, pretrained=False, head_type=head_type)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, img_size


def predict_batch_with_tta(model, x, tta=True):
    """Инференс модели с Test-Time Augmentation (горизонтальный флип)."""
    p = model(x)[0]
    if tta:
        x_flip = torch.flip(x, dims=[-1])
        p_flip = model(x_flip)[0]
        p = 0.5 * (p + p_flip)
    return p.detach().cpu().numpy()


def predict_single_model(model, img_paths, img_size, device, batch_size=16, tta=True):
    """Получает предсказания одной модели для списка путей к изображениям."""
    transform = build_transforms(train=False, img_size=img_size)
    predictions = []

    with torch.no_grad():
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i : i + batch_size]
            batch_tensors = []
            for p in batch_paths:
                img = read_rgb_image(p)
                batch_tensors.append(transform(image=img)["image"])
            x = torch.stack(batch_tensors).to(device)
            preds = predict_batch_with_tta(model, x, tta=tta)
            predictions.extend(preds.tolist())

    return np.asarray(predictions, dtype=np.float64)


def compute_oof_predictions(folds_df, images_dir, cnn_models_dir, device, batch_size=16, tta=True):
    """Вычисляет Out-Of-Fold предсказания для всех фолдов."""
    images_dir = Path(images_dir)
    cnn_models_dir = Path(cnn_models_dir)
    
    oof_preds = np.zeros(len(folds_df), dtype=np.float64)
    unique_folds = sorted(folds_df["fold"].unique())
    print(f"Вычисление OOF-предсказаний по {len(unique_folds)} фолдам...", flush=True)

    for fold in unique_folds:
        ckpt_path = cnn_models_dir / f"fold{fold}_best.pt"
        if not ckpt_path.is_file():
            # Попробуем найти любой подходящий чекпоинт
            candidates = list(cnn_models_dir.glob(f"*fold*{fold}*.pt"))
            if candidates:
                ckpt_path = candidates[0]
            else:
                raise FileNotFoundError(f"Не найден чекпоинт фолда {fold}: {ckpt_path}")

        print(f"  [Fold {fold}] Загрузка {ckpt_path.name}...", flush=True)
        model, img_size = load_cnn_model(ckpt_path, device)

        fold_mask = folds_df["fold"] == fold
        fold_indices = np.where(fold_mask)[0]
        fold_rows = folds_df.iloc[fold_indices]

        img_paths = [images_dir / f"{img_id}.jpg" for img_id in fold_rows["image_id"]]
        preds = predict_single_model(model, img_paths, img_size, device, batch_size=batch_size, tta=tta)
        oof_preds[fold_indices] = preds

        fold_mae = float(np.mean(np.abs(fold_rows["load_pct"].values - preds)))
        print(f"  [Fold {fold}] MAE = {fold_mae:.4f}", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    overall_mae = float(np.mean(np.abs(folds_df["load_pct"].values - oof_preds)))
    print(f"-> Итоговый OOF MAE ансамбля CNN: {overall_mae:.4f} п.п.", flush=True)
    return oof_preds


def predict_ensemble_test(checkpoints, test_rows, images_dir, device, batch_size=16, tta=True):
    """Инференс ансамбля чекпоинтов на тестовой выборке с усреднением."""
    images_dir = Path(images_dir)
    img_paths = [images_dir / f"{r['image_id']}.jpg" for r in test_rows]

    all_model_preds = []
    print(f"Инференс ансамбля CNN ({len(checkpoints)} моделей) на тестовой выборке...", flush=True)

    for i, ckpt_path in enumerate(checkpoints):
        ckpt_path = Path(ckpt_path)
        print(f"  [{i+1}/{len(checkpoints)}] Прогон модели {ckpt_path.name}...", flush=True)
        model, img_size = load_cnn_model(ckpt_path, device)
        preds = predict_single_model(model, img_paths, img_size, device, batch_size=batch_size, tta=tta)
        all_model_preds.append(preds)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_preds = np.mean(all_model_preds, axis=0)
    return mean_preds


def calibrate_predictions(values, lower=3.5, upper=94.5):
    """Калибровка экстремальных значений загрузки."""
    vals = np.asarray(values, dtype=np.float64).copy()
    vals[vals <= lower] = 0.0
    vals[vals >= upper] = 100.0
    return np.clip(vals, 0.0, 100.0)


def extract_features_table(rows, images_dir, contract, cache_dir):
    """Извлекает и кэширует 349 геометрических признаков."""
    print(f"Извлечение/загрузка из кэша 349 признаков для {len(rows)} объектов...", flush=True)
    features = cached_features(images_dir, rows, contract, cache_dir=cache_dir)
    if features.shape != (len(rows), TOTAL_FEATURE_DIM):
        raise ValueError(f"Ожидалась размерность [{len(rows)}, {TOTAL_FEATURE_DIM}], получено {features.shape}")
    return features


def train_stacking(args):
    """Обучение стэкинга (4-фолдовый CNN + 349 геометрических признаков -> Ridge)."""
    device = get_device(args.device)
    print(f"=== ЗАПУСК ОБУЧЕНИЯ СТЭКИНГА (устройство: {device}) ===", flush=True)

    folds_path = Path(getattr(args, "folds", "./DataSet/train/folds.csv"))
    if not folds_path.is_file():
        raise FileNotFoundError(f"Файл фолдов не найден: {folds_path}")

    folds_df = pd.read_csv(folds_path)
    train_rows = folds_df.to_dict(orient="records")
    y_true = folds_df["load_pct"].values.astype(np.float64)

    # 1. Проверяем наличие чекпоинтов CNN
    cnn_dir = Path(getattr(args, "cnn_models_dir", "./models/cnn_v1"))
    ckpts = sorted(list(cnn_dir.glob("fold*_best.pt")))
    if len(ckpts) < 4:
        # Если чекпоинтов нет, пробуем запустить обучение CNN
        print(f"Внимание: в {cnn_dir} найдено {len(ckpts)}/4 чекпоинтов. Запускаем обучение CNN...", flush=True)
        from CNN.train_cnn import train_from_baseline
        train_from_baseline(args)
        ckpts = sorted(list(cnn_dir.glob("fold*_best.pt")))
        if len(ckpts) < 4:
            raise FileNotFoundError(f"Не удалось обучить/найти все 4 фолда в {cnn_dir}")

    # 2. Вычисляем честные OOF-предсказания CNN
    oof_cnn = compute_oof_predictions(
        folds_df=folds_df,
        images_dir=args.images,
        cnn_models_dir=cnn_dir,
        device=device,
        batch_size=args.batch_size,
        tta=getattr(args, "tta", True),
    )

    cnn_raw_mae = float(np.mean(np.abs(y_true - oof_cnn)))
    cnn_calib_mae = float(np.mean(np.abs(y_true - calibrate_predictions(oof_cnn))))
    print(f"CNN OOF MAE: {cnn_raw_mae:.4f} (с калибровкой: {cnn_calib_mae:.4f})", flush=True)

    # 3. Извлекаем 349 геометрических признаков
    contract = build_contract(
        truck_weights=args.truck_weights,
        floor_weights=args.floor_weights,
        ceiling_weights=getattr(args, "ceiling_weights", None),
    )
    cache_dir = getattr(args, "feature_cache", None) or Path(args.model).parent / "feature_cache"
    geo_features = extract_features_table(train_rows, args.images, contract, cache_dir)

    # 4. Формируем мета-матрицу признаков: [cnn_oof, 349_geo_features]
    X_meta = np.column_stack([oof_cnn, geo_features])
    print(f"Мета-матрица признаков сформирована: {X_meta.shape}", flush=True)

    # 5. Обучаем Ridge мета-модель со стандартизацией
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_meta)

    alphas = [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0]
    ridge = RidgeCV(alphas=alphas)
    ridge.fit(X_scaled, y_true)

    oof_meta_pred = np.clip(ridge.predict(X_scaled), 0.0, 100.0)
    meta_mae = float(np.mean(np.abs(y_true - oof_meta_pred)))

    oof_final_pred = calibrate_predictions(oof_meta_pred)
    final_mae = float(np.mean(np.abs(y_true - oof_final_pred)))
    within_10 = float(100.0 * np.mean(np.abs(y_true - oof_final_pred) <= 10.0))

    print("=" * 60, flush=True)
    print(f"🏆 РЕЗУЛЬТАТЫ СТЭКИНГА:", flush=True)
    print(f"  Выбранный alpha Ridge:   {ridge.alpha_}", flush=True)
    print(f"  Исходный CNN OOF MAE:    {cnn_raw_mae:.4f} п.п.", flush=True)
    print(f"  Стэкинг (+ геометрия) MAE:{meta_mae:.4f} п.п.", flush=True)
    print(f"  Финальный MAE (калибр.):  {final_mae:.4f} п.п.", flush=True)
    print(f"  Точность within 10%:     {within_10:.2f}%", flush=True)
    print("=" * 60, flush=True)

    # Сохраняем ошибки валидации
    errors_path = getattr(args, "errors_output", "")
    if errors_path:
        errors_df = pd.DataFrame({
            "image_id": folds_df["image_id"],
            "fold": folds_df["fold"],
            "actual_load_pct": y_true,
            "cnn_pred": oof_cnn,
            "stacked_pred": oof_final_pred,
            "absolute_error_pct": np.abs(y_true - oof_final_pred),
        }).sort_values("absolute_error_pct", ascending=False)
        Path(errors_path).parent.mkdir(parents=True, exist_ok=True)
        errors_df.to_csv(errors_path, index=False)
        print(f"Ошибки валидации сохранены: {errors_path}", flush=True)

    # 6. Сохраняем артефакты стэкинга
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "method": "stacking",
        "scaler": scaler,
        "ridge": ridge,
        "cnn_checkpoints": [str(p) for p in ckpts],
        "feature_contract": contract,
        "dimension": TOTAL_FEATURE_DIM,
    }
    with open(model_path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(payload, f)

    report = {
        "selected_method": "stacking",
        "cnn_models_count": len(ckpts),
        "checkpoints": [str(p) for p in ckpts],
        "features_dimension": TOTAL_FEATURE_DIM,
        "best_alpha": float(ridge.alpha_),
        "cnn_raw_oof_mae": cnn_raw_mae,
        "cnn_calibrated_oof_mae": cnn_calib_mae,
        "stacking_oof_mae": meta_mae,
        "final_calibrated_oof_mae": final_mae,
        "within_10_pct": within_10,
        "feature_contract": contract,
    }
    report_path = model_path.with_suffix(".json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Мета-модель сохранена: {model_path.with_suffix('.pkl')}", flush=True)
    print(f"Отчёт сохранён:        {report_path}", flush=True)


def predict_stacking(args):
    """Инференс стэкинг-пайплайна на тестовых данных."""
    device = get_device(getattr(args, "device", "auto"))
    print(f"=== ЗАПУСК ИНФЕРЕНСА СТЭКИНГА (устройство: {device}) ===", flush=True)

    model_path = Path(args.model)
    pkl_path = model_path.with_suffix(".pkl") if model_path.suffix != ".pkl" else model_path
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Файл мета-модели не найден: {pkl_path}")

    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    scaler = payload["scaler"]
    ridge = payload["ridge"]
    contract = payload["feature_contract"]
    ckpts = [resolve_saved_path(p) for p in payload["cnn_checkpoints"]]

    test_rows = read_test_table(args.test_csv)
    print(f"Загружено {len(test_rows)} тестовых объектов из {args.test_csv}", flush=True)

    # 1. Предсказания 4-фолдового ансамбля CNN
    cnn_test_preds = predict_ensemble_test(
        checkpoints=ckpts,
        test_rows=test_rows,
        images_dir=args.images,
        device=device,
        batch_size=getattr(args, "batch_size", 16),
        tta=getattr(args, "tta", True),
    )

    # 2. Извлечение 349 геометрических признаков для теста
    cache_dir = getattr(args, "feature_cache", None) or Path(args.output).parent / "feature_cache"
    geo_test = extract_features_table(test_rows, args.images, contract, cache_dir)

    # 3. Мета-предсказание
    X_test_meta = np.column_stack([cnn_test_preds, geo_test])
    X_test_scaled = scaler.transform(X_test_meta)

    raw_preds = np.clip(ridge.predict(X_test_scaled), 0.0, 100.0)

    # 4. Калибровка
    if getattr(args, "calibrate", True):
        final_preds = calibrate_predictions(raw_preds)
    else:
        final_preds = raw_preds

    write_submission(test_rows, final_preds, args.output)
    print(f"🏆 Сабмит успешно записан в: {args.output} ({len(final_preds)} строк)", flush=True)

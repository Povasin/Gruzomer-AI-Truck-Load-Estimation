import json
from pathlib import Path
import pickle
import numpy as np
import torch

from report import read_test_table
from submission import write_submission


def predict_cnn_ensemble(model_path, rows, images_dir, device="auto", batch_size=16, tta=True, calibrate=True):
    from CNN.model import TruckLoadNet
    from CNN.dataset import build_transforms, read_rgb_image

    model_path = Path(model_path)
    if model_path.is_dir():
        checkpoints = sorted(list(model_path.glob("fold*_best.pt")))
        if not checkpoints:
            checkpoints = sorted(list(model_path.glob("*.pt")))
    elif model_path.suffix.lower() == ".pt":
        checkpoints = [model_path]
    elif model_path.suffix.lower() == ".json":
        with open(model_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "checkpoints" in cfg:
            checkpoints = [Path(p) for p in cfg["checkpoints"]]
        elif "output_dir" in cfg:
            checkpoints = sorted(list(Path(cfg["output_dir"]).glob("fold*_best.pt")))
        elif "models_dir" in cfg:
            checkpoints = sorted(list(Path(cfg["models_dir"]).glob("fold*_best.pt")))
        else:
            checkpoints = sorted(list(model_path.parent.glob("fold*_best.pt")))
    elif model_path.suffix.lower() == ".npz":
        with np.load(model_path, allow_pickle=False) as saved:
            if "checkpoints" in saved:
                checkpoints = [Path(str(p)) for p in saved["checkpoints"]]
            else:
                checkpoints = sorted(list(model_path.parent.glob("fold*_best.pt")))
    else:
        raise ValueError(f"Неизвестный формат модели CNN: {model_path}")

    if not checkpoints:
        raise FileNotFoundError(f"Не найдены чекпоинты моделей по пути {model_path}")

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    models = []
    img_size = 320
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        backbone = ckpt.get("backbone", "resnet18")
        state = ckpt.get("model_state_dict", {})
        head_type = ckpt.get("head_type", "distributional" if "dist_head.weight" in state else "scalar")
        img_size = ckpt.get("img_size", img_size)
        m = TruckLoadNet(backbone_name=backbone, pretrained=False, head_type=head_type)
        m.load_state_dict(state, strict=False)
        m.to(dev).eval()
        models.append(m)

    transform = build_transforms(train=False, img_size=img_size)
    predictions = []

    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i:i + batch_size]
            batch_tensors = []
            for r in batch_rows:
                img_file = Path(images_dir) / f"{r['image_id']}.jpg"
                img = read_rgb_image(img_file)
                batch_tensors.append(transform(image=img)["image"])
            x = torch.stack(batch_tensors).to(dev)

            model_batch_preds = []
            for m in models:
                p = m(x)[0]
                if tta:
                    x_flip = torch.flip(x, dims=[-1])
                    p_flip = m(x_flip)[0]
                    p = 0.5 * (p + p_flip)
                model_batch_preds.append(p.cpu().numpy())

            batch_mean = np.mean(model_batch_preds, axis=0)
            predictions.extend(batch_mean.tolist())

    values = np.asarray(predictions, dtype=np.float64)

    if calibrate:
        values[values >= 94.5] = 100.0
        values[values <= 3.5] = 0.0

    return np.clip(values, 0.0, 100.0)


def predict(args):
    model_path = Path(args.model)
    rows = read_test_table(args.test_csv)

    # Проверка на CNN чекпоинт или директорию ансамбля
    if model_path.is_dir() or model_path.suffix.lower() == ".pt":
        values = predict_cnn_ensemble(
            model_path,
            rows,
            args.images,
            device=getattr(args, "device", "auto"),
            batch_size=getattr(args, "batch_size", 16),
            tta=getattr(args, "tta", True),
            calibrate=getattr(args, "calibrate", True),
        )
        write_submission(rows, values, args.output)
        return

    # Загружаем JSON конфигурацию
    if model_path.suffix.lower() == ".json":
        with open(model_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        method = meta.get("selected_method") or meta.get("method")
        if method == "stacking":
            from stacking import predict_stacking
            return predict_stacking(args)
        if method in ("cnn", "cnn_ensemble") or "folds" in meta:
            values = predict_cnn_ensemble(
                model_path,
                rows,
                args.images,
                device=getattr(args, "device", "auto"),
                batch_size=getattr(args, "batch_size", 16),
                tta=getattr(args, "tta", True),
                calibrate=getattr(args, "calibrate", True),
            )
            write_submission(rows, values, args.output)
            return

    # Загружаем ровно указанный файл: соседний .pkl не подменяет .npz.
    if model_path.suffix.lower() == ".pkl":
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        method = data.get("method")
        if method == "stacking":
            from stacking import predict_stacking
            return predict_stacking(args)
        model = data.get("model")
    else:
        with np.load(model_path, allow_pickle=False) as saved:
            model = {key: saved[key] for key in saved.files}
        method = str(model.get("method", ""))

    if method == "stacking":
        from stacking import predict_stacking
        return predict_stacking(args)
    elif method == "hybrid":
        from hybrid.hybrid import predict as predict_hybrid
        return predict_hybrid(args)
    elif method in ("cnn", "cnn_ensemble"):
        values = predict_cnn_ensemble(
            model_path,
            rows,
            args.images,
            device=getattr(args, "device", "auto"),
            batch_size=getattr(args, "batch_size", 16),
            tta=getattr(args, "tta", True),
            calibrate=getattr(args, "calibrate", True),
        )
    elif method == "median":
        values = np.full(len(rows), float(model["median"]))
    elif method == "boosting":
        from manual.features import features
        x_test = features(args.images, rows)
        values = np.clip(model.predict(x_test), 0, 100)
    elif method == "ridge":
        from train import predict_ridge
        from manual.features import features
        x_test = features(args.images, rows)
        values = predict_ridge(model, x_test)
    else:
        raise ValueError(f"Unknown method: {method}")

    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictions")

    write_submission(rows, values, args.output)
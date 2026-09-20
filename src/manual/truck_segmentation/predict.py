from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# CONFIG
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_DIR
    / "src"
    / "segmentation"
    / "models"
    / "best_unet_resnet18.pth"
)


# ============================================================
# MODEL
# ============================================================

def create_model():
    """
    Та же архитектура, которая использовалась при обучении.
    """

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,  # pretrained веса уже находятся в checkpoint
        in_channels=3,
        classes=1,
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path, device):
    print(f"Loading model: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Размер изображения сохранялся в checkpoint во время обучения
    image_size = checkpoint.get(
        "image_size",
        512,
    )

    model = create_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(
        f"Model loaded | "
        f"image_size={image_size} | "
        f"device={device}"
    )

    if "val_iou" in checkpoint:
        print(
            f"Checkpoint IoU: "
            f"{checkpoint['val_iou']:.4f}"
        )

    if "val_dice" in checkpoint:
        print(
            f"Checkpoint Dice: "
            f"{checkpoint['val_dice']:.4f}"
        )

    return model, image_size


# ============================================================
# PREPROCESS
# ============================================================

def create_transform(image_size):
    """
    ВАЖНО:
    Normalize должен быть таким же,
    как при обучении.
    """

    return A.Compose([
        A.Resize(
            image_size,
            image_size,
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        ToTensorV2(),
    ])


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

def predict_image(
    model,
    image_path,
    transform,
    device,
    threshold=0.5,
):
    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:
        raise RuntimeError(
            f"Не удалось открыть изображение: {image_path}"
        )

    original_height, original_width = \
        image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    transformed = transform(
        image=image_rgb
    )

    tensor = transformed["image"]

    # C,H,W -> 1,C,H,W
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.sigmoid(
            logits
        )

    # 1,1,H,W -> H,W
    probability_mask = (
        probabilities[0, 0]
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # RESIZE MASK BACK TO ORIGINAL IMAGE
    # --------------------------------------------------------

    probability_mask = cv2.resize(
        probability_mask,
        (
            original_width,
            original_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # --------------------------------------------------------
    # BINARY MASK
    # --------------------------------------------------------

    binary_mask = (
        probability_mask >= threshold
    ).astype(np.uint8)

    # Для сохранения PNG:
    # 0 / 1 -> 0 / 255
    mask_255 = binary_mask * 255

    # --------------------------------------------------------
    # CLEAN IMAGE
    # --------------------------------------------------------

    cleaned = image_bgr.copy()

    cleaned[
        binary_mask == 0
    ] = 0

    return (
        mask_255,
        cleaned,
        probability_mask,
    )


# ============================================================
# PREVIEW
# ============================================================

def create_preview(
    original,
    mask,
    cleaned,
):
    """
    Создает:
    ORIGINAL | MASK | CLEANED
    """

    height, width = original.shape[:2]

    mask_bgr = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR,
    )

    # Подписи
    original_view = original.copy()
    mask_view = mask_bgr.copy()
    cleaned_view = cleaned.copy()

    cv2.putText(
        original_view,
        "ORIGINAL",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        mask_view,
        "PREDICTED MASK",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        cleaned_view,
        "CLEANED",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    return np.hstack([
        original_view,
        mask_view,
        cleaned_view,
    ])


# ============================================================
# GET IMAGES
# ============================================================

def get_images(input_path):

    input_path = Path(input_path)

    # Один файл
    if input_path.is_file():

        return [input_path]

    # Папка
    if input_path.is_dir():

        images = sorted([
            path
            for path in input_path.iterdir()
            if path.suffix.lower()
            in VALID_EXTENSIONS
        ])

        return images

    raise FileNotFoundError(
        f"Путь не существует: {input_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Изображение или папка с изображениями",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Путь к checkpoint модели",
    )

    parser.add_argument(
        "--output",
        default="segmentation_predictions",
        help="Куда сохранить результаты",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold бинаризации mask",
    )

    args = parser.parse_args()


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("DEVICE:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # ========================================================
    # OUTPUT DIRS
    # ========================================================

    output_dir = Path(args.output)

    masks_dir = \
        output_dir / "masks"

    cleaned_dir = \
        output_dir / "cleaned"

    previews_dir = \
        output_dir / "previews"

    probability_dir = \
        output_dir / "probabilities"


    for directory in [
        masks_dir,
        cleaned_dir,
        previews_dir,
        probability_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


    # ========================================================
    # MODEL
    # ========================================================

    model, image_size = load_model(
        args.model,
        device,
    )

    transform = create_transform(
        image_size
    )


    # ========================================================
    # IMAGES
    # ========================================================

    images = get_images(
        args.input
    )

    print(
        f"Images found: {len(images)}"
    )


    # ========================================================
    # PREDICTION LOOP
    # ========================================================

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        print(
            f"[{index}/{len(images)}] "
            f"{image_path.name}"
        )

        try:

            mask, cleaned, probability = \
                predict_image(
                    model=model,
                    image_path=image_path,
                    transform=transform,
                    device=device,
                    threshold=args.threshold,
                )


            # ------------------------------------------------
            # ORIGINAL
            # ------------------------------------------------

            original = cv2.imread(
                str(image_path)
            )


            # ------------------------------------------------
            # SAVE MASK
            # ------------------------------------------------

            mask_path = (
                masks_dir
                / f"{image_path.stem}.png"
            )

            cv2.imwrite(
                str(mask_path),
                mask,
            )


            # ------------------------------------------------
            # SAVE CLEANED IMAGE
            # ------------------------------------------------

            cleaned_path = (
                cleaned_dir
                / f"{image_path.stem}.jpg"
            )

            cv2.imwrite(
                str(cleaned_path),
                cleaned,
            )


            # ------------------------------------------------
            # SAVE PREVIEW
            # ------------------------------------------------

            preview = create_preview(
                original,
                mask,
                cleaned,
            )

            preview_path = (
                previews_dir
                / f"{image_path.stem}_preview.jpg"
            )

            cv2.imwrite(
                str(preview_path),
                preview,
            )


            # ------------------------------------------------
            # SAVE PROBABILITY MASK
            # ------------------------------------------------

            probability_255 = (
                probability * 255
            ).clip(
                0,
                255,
            ).astype(
                np.uint8
            )

            probability_path = (
                probability_dir
                / f"{image_path.stem}.png"
            )

            cv2.imwrite(
                str(probability_path),
                probability_255,
            )


        except Exception as error:

            print(
                f"[ERROR] "
                f"{image_path.name}: "
                f"{error}"
            )


    print()
    print("==============================")
    print("Prediction finished")
    print("==============================")

    print(
        f"Masks:       {masks_dir}"
    )

    print(
        f"Cleaned:     {cleaned_dir}"
    )

    print(
        f"Previews:    {previews_dir}"
    )

    print(
        f"Probabilities: {probability_dir}"
    )


if __name__ == "__main__":
    main()

# python .\src\segmentation\predict.py --input .\DataSet\train\images\img_18e1e7031747b0aedf08.jpg `
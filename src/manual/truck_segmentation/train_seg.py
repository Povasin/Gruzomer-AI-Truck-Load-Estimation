from pathlib import Path
import random
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_DIR / "DataSet" / "train" / "images"
MASK_DIR = PROJECT_DIR / "debug_output" / "masks"

IMAGE_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 15
LR = 1e-4

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("DEVICE:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA недоступна. Обучение идет на CPU.")
MODEL_PATH = (
    PROJECT_DIR
    / "src"
    / "segmentation"
    / "models"
    / "best_unet_resnet18.pth"
)


# ============================================================
# SEED
# ============================================================

print("DEVICE:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# COLLECT FILES
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def collect_samples():

    samples = []

    for image_path in IMAGE_DIR.iterdir():

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        mask_path = MASK_DIR / f"{image_path.stem}.png"

        if not mask_path.exists():

            print(
                f"[WARNING] Нет mask для "
                f"{image_path.name}"
            )

            continue

        samples.append(
            (
                image_path,
                mask_path
            )
        )

    return sorted(samples)


samples = collect_samples()

print(
    f"Найдено размеченных изображений: "
    f"{len(samples)}"
)


# ============================================================
# SPLIT
# ============================================================

train_samples, val_samples = train_test_split(
    samples,
    test_size=0.20,
    random_state=SEED
)

print(
    f"Train: {len(train_samples)}"
)

print(
    f"Validation: {len(val_samples)}"
)


# ============================================================
# AUGMENTATIONS
# ============================================================

train_transform = A.Compose([

    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    A.HorizontalFlip(
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.4
    ),

    A.GaussNoise(
        p=0.2
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])


val_transform = A.Compose([

    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])


# ============================================================
# DATASET
# ============================================================

class TruckDataset(Dataset):

    def __init__(
        self,
        samples,
        transform
    ):

        self.samples = samples
        self.transform = transform


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        image_path, mask_path = \
            self.samples[index]

        # -----------------------------
        # IMAGE
        # -----------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                f"Не удалось открыть "
                f"{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # -----------------------------
        # MASK
        # -----------------------------

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            raise RuntimeError(
                f"Не удалось открыть "
                f"{mask_path}"
            )


        # 0 / 255
        # превращаем в 0 / 1

        mask = (
            mask > 127
        ).astype(
            np.float32
        )


        # -----------------------------
        # TRANSFORM
        # -----------------------------

        transformed = self.transform(
            image=image,
            mask=mask
        )

        image = transformed["image"]

        mask = transformed["mask"]


        # H x W
        # →
        # 1 x H x W

        mask = mask.unsqueeze(0).float()


        return image, mask


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = TruckDataset(
    train_samples,
    train_transform
)

val_dataset = TruckDataset(
    val_samples,
    val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = smp.Unet(

    encoder_name="resnet18",

    # pretrained ImageNet
    encoder_weights="imagenet",

    in_channels=3,

    # бинарная segmentation
    classes=1

).to(DEVICE)


# ============================================================
# LOSS
# ============================================================

bce_loss = torch.nn.BCEWithLogitsLoss()

dice_loss = smp.losses.DiceLoss(
    mode="binary",
    from_logits=True
)


def loss_fn(
    prediction,
    target
):

    return (
        bce_loss(
            prediction,
            target
        )
        +
        dice_loss(
            prediction,
            target
        )
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    targets
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities > 0.5
    ).float()


    # flatten
    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )


    intersection = (
        predictions * targets
    ).sum(dim=1)


    union = (
        predictions
        +
        targets
        -
        predictions * targets
    ).sum(dim=1)


    # -------------------------
    # IoU
    # -------------------------

    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )


    # -------------------------
    # Dice
    # -------------------------

    dice = (
        2 * intersection + 1e-7
    ) / (
        predictions.sum(dim=1)
        +
        targets.sum(dim=1)
        +
        1e-7
    )


    return (
        iou.mean().item(),
        dice.mean().item()
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=1e-4
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=3
)


# ============================================================
# TRAIN
# ============================================================

best_val_loss = float("inf")

print("Пробуем получить один batch...")

images, masks = next(iter(train_loader))

print("Batch получен")
print("images:", images.shape)
print("masks:", masks.shape)

images = images.to(DEVICE)
masks = masks.to(DEVICE)

print("Отправили на device")

logits = model(images)

print("Forward прошёл")
print("logits:", logits.shape)
for epoch in range(
    1,
    EPOCHS + 1
):

    # ========================================================
    # TRAIN
    # ========================================================
    print(f"\n===== EPOCH {epoch}/{EPOCHS} =====")
    model.train()

    train_loss = 0

    train_bar = tqdm(
        train_loader,
        desc="Train"
    )
    for images, masks in train_loader:

        images = images.to(DEVICE)

        masks = masks.to(DEVICE)


        optimizer.zero_grad()


        logits = model(
            images
        )


        loss = loss_fn(
            logits,
            masks
        )


        loss.backward()


        optimizer.step()


        train_loss += \
            loss.item()


    train_loss /= \
        len(train_loader)


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0

    total_iou = 0

    total_dice = 0

    batches = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)


            logits = model(
                images
            )


            loss = loss_fn(
                logits,
                masks
            )


            val_loss += \
                loss.item()


            iou, dice = calculate_metrics(
                logits,
                masks
            )


            total_iou += iou

            total_dice += dice

            batches += 1


    val_loss /= \
        len(val_loader)


    val_iou = \
        total_iou / batches

    val_dice = \
        total_dice / batches


    scheduler.step(
        val_loss
    )


    # ========================================================
    # LOG
    # ========================================================

    current_lr = \
        optimizer.param_groups[0]["lr"]


    print(
        f"\nEpoch "
        f"{epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Val Loss:   "
        f"{val_loss:.4f}"
    )

    print(
        f"Val IoU:    "
        f"{val_iou:.4f}"
    )

    print(
        f"Val Dice:   "
        f"{val_dice:.4f}"
    )

    print(
        f"LR:         "
        f"{current_lr:.6f}"
    )


    # ========================================================
    # SAVE BEST
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "val_loss":
                    val_loss,

                "val_iou":
                    val_iou,

                "val_dice":
                    val_dice,

                "image_size":
                    IMAGE_SIZE
            },

            MODEL_PATH
        )


        print(
            f"BEST MODEL SAVED → "
            f"{MODEL_PATH}"
        )


print()
print("Training finished.")
print(
    f"Best model: {MODEL_PATH}"
)
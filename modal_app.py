import os
import modal

app = modal.App("roi-floor-roof-api")

MODEL_URL = (
    "https://huggingface.co/"
    "KirillTron22/roi-floor-roof-model/"
    "resolve/main/convnext_final.pt"
)

MODEL_PATH = "/models/convnext_final.pt"


image = (
    modal.Image.debian_slim(python_version="3.11")

    # CPU-only PyTorch. Не тащим CUDA/NVIDIA пакеты.
    .pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )

    .pip_install(
        "fastapi==0.141.1",
        "uvicorn==0.53.0",
        "python-multipart==0.0.32",
        "pillow==12.3.0",
        "numpy>=2.0",
        "opencv-python-headless>=4.10",
        "timm>=1.0",
        "albumentations>=2.0",
    )

    # Скачиваем модель один раз во время build.
    .run_commands(
        "mkdir -p /models",
        (
            "python -c \""
            "import urllib.request; "
            f"urllib.request.urlretrieve('{MODEL_URL}', '{MODEL_PATH}')"
            "\""
        ),
    )

    # Код backend.
    .add_local_dir(
        "backend",
        remote_path="/root/backend",
    )

    # CV-код. Dataset и локальные веса не загружаем.
    .add_local_dir(
        "CV",
        remote_path="/root/CV",
        ignore=[
            "DataSet/**",
            "models/**",
            "**/__pycache__/**",
            "**/*.ipynb",
        ],
    )
)


@app.function(
    image=image,
    memory=3072,
    cpu=1,
    timeout=300,
    startup_timeout=300,
)
@modal.asgi_app()
def api():
    os.environ["CV_MODEL_PATH"] = MODEL_PATH
    os.environ["CV_DEVICE"] = "cpu"

    # Для MVP отключаем TTA, иначе CPU inference будет заметно дольше.
    os.environ["CV_USE_TTA"] = "false"
    os.environ["CV_USE_CLAHE_TTA"] = "false"

    os.environ["FRONTEND_ORIGINS"] = (
        "https://roi-floor-roof-seg.vercel.app"
    )

    from backend.app.main import app as fastapi_app

    return fastapi_app
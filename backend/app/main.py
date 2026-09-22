from __future__ import annotations

import os
from xml.sax.saxutils import escape

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError

from .estimator import estimate_load
from .repository import MemoryPredictionRepository

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
repository = MemoryPredictionRepository()

app = FastAPI(title="Load Vision API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "FRONTEND_ORIGINS", "https://roi-floor-roof-seg.vercel.app"
    ).split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def clean_transport_id(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="Введите номер перевозки.")
    if len(cleaned) > 80:
        raise HTTPException(status_code=422, detail="Номер перевозки слишком длинный.")
    return cleaned


def prediction_response(prediction, wants_xml: bool, include_method: bool = False, status_code: int = 200):
    payload = {**prediction.__dict__, "status": "success"}
    if include_method:
        payload["method"] = "cv_pipeline"
    if not wants_xml:
        return payload
    xml = "".join(
        f"<{key}>{escape(str(value))}</{key}>" for key, value in payload.items()
    )
    return Response(
        content=f"<?xml version=\"1.0\" encoding=\"UTF-8\"?><prediction>{xml}</prediction>",
        media_type="application/xml",
        status_code=status_code,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/predictions", status_code=status.HTTP_201_CREATED)
async def create_prediction(
    request: Request, transport_id: str = Form(...), image: UploadFile = File(...)
):
    transport_id = clean_transport_id(transport_id)
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Поддерживаются изображения JPG, PNG и WEBP.")

    content = await image.read()
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=422, detail="Добавьте фотографию грузового отсека до 10 МБ.")
    try:
        prediction = estimate_load(content)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="Не удалось обработать изображение. Попробуйте загрузить другой файл.",
        ) from None
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Модель оценки временно недоступна.",
        ) from None

    result = repository.save(
        transport_id,
        load_pct=float(prediction["load_pct"]),
        cargo_type=str(prediction.get("cargo_type", "unknown")),
        models_count=int(prediction.get("models_count", 0)),
        backbone=str(prediction.get("backbone", "unknown")),
    )
    return prediction_response(
        result,
        "application/xml" in request.headers.get("accept", ""),
        include_method=True,
        status_code=status.HTTP_201_CREATED,
    )


@app.get("/api/v1/predictions/{transport_id}")
def get_prediction(transport_id: str, request: Request):
    transport_id = clean_transport_id(transport_id)
    result = repository.get(transport_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Результат для перевозки {transport_id} не найден.",
        )
    return prediction_response(result, "application/xml" in request.headers.get("accept", ""))

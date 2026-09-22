from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.repository import MemoryPredictionRepository


def image_payload() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (24, 24), (80, 80, 80)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_prediction_can_be_returned_as_xml(monkeypatch):
    monkeypatch.setattr(
        main,
        "estimate_load",
        lambda _image: {
            "load_pct": 67,
            "cargo_type": "boxes",
            "models_count": 1,
            "backbone": "convnext_tiny",
        },
    )
    monkeypatch.setattr(main, "repository", MemoryPredictionRepository())
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/predictions",
        headers={"Accept": "application/xml"},
        data={"transport_id": "TR-XML-1"},
        files={"image": ("truck.jpg", image_payload(), "image/jpeg")},
    )

    assert response.status_code == 201
    assert response.headers["content-type"].startswith("application/xml")
    assert "<transport_id>TR-XML-1</transport_id>" in response.text
    assert "<method>cv_pipeline</method>" in response.text

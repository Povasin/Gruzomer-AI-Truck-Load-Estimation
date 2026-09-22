from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app import main


def test_api_returns_a_safe_error_when_the_cv_model_is_incompatible(monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (24, 24), "white").save(buffer, format="JPEG")
    monkeypatch.setattr(main, "estimate_load", lambda _image: (_ for _ in ()).throw(RuntimeError("feature contract mismatch")))

    response = TestClient(main.app, raise_server_exceptions=False).post(
        "/api/v1/predictions",
        data={"transport_id": "TR-CV-1"},
        files={"image": ("truck.jpg", buffer.getvalue(), "image/jpeg")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Модель оценки временно недоступна."

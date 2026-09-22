from io import BytesIO

from PIL import Image

from app import estimator
from app.estimator import estimate_load


def make_image(brightness: int) -> bytes:
    image = Image.new("RGB", (48, 32), (brightness, brightness, brightness))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_estimate_load_returns_a_bounded_percentage_for_a_valid_photo(monkeypatch):
    monkeypatch.setattr(estimator, "load_one_photo", lambda *_args, **_kwargs: 42.4, raising=False)

    result = estimate_load(make_image(100))

    assert 0 <= result <= 100


def test_estimate_load_is_deterministic_for_the_same_photo(monkeypatch):
    monkeypatch.setattr(estimator, "load_one_photo", lambda *_args, **_kwargs: 55.0, raising=False)
    photo = make_image(180)

    assert estimate_load(photo) == estimate_load(photo)


def test_estimate_load_delegates_a_photo_to_the_cv_pipeline(monkeypatch):
    calls = []

    def fake_load_one_photo(path, model_path, **options):
        calls.append((path, model_path, options))
        return 67.25

    monkeypatch.setattr(estimator, "load_one_photo", fake_load_one_photo, raising=False)

    assert estimate_load(make_image(120)) == 67
    assert calls[0][2]["method"] == "auto"


def test_estimate_load_marks_a_model_failure_as_a_service_error(monkeypatch):
    monkeypatch.setattr(estimator, "load_one_photo", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("contract mismatch")), raising=False)

    with pytest.raises(RuntimeError, match="CV pipeline"):
        estimate_load(make_image(120))

from types import SimpleNamespace

from app.repository import SupabasePredictionRepository


class FakePredictionsTable:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.selected_transport_id: str | None = None

    def upsert(self, payload: dict[str, object], on_conflict: str):
        assert on_conflict == "transport_id"
        self.rows[str(payload["transport_id"])] = payload
        return self

    def select(self, _columns: str):
        return self

    def eq(self, column: str, value: str):
        assert column == "transport_id"
        self.selected_transport_id = value
        return self

    def maybe_single(self):
        return self

    def execute(self):
        if self.selected_transport_id is not None:
            return SimpleNamespace(data=self.rows.get(self.selected_transport_id))
        return SimpleNamespace(data=None)


class FakeSupabaseClient:
    def __init__(self) -> None:
        self.predictions = FakePredictionsTable()

    def table(self, name: str) -> FakePredictionsTable:
        assert name == "predictions"
        return self.predictions


def test_saves_a_prediction_and_returns_it_by_transport_id():
    client = FakeSupabaseClient()
    repository = SupabasePredictionRepository(client)

    saved = repository.save(
        "TR-42",
        load_pct=37.5,
        cargo_type="pallets",
        models_count=1,
        backbone="convnext_tiny",
    )

    assert saved.transport_id == "TR-42"
    assert saved.load_pct == 37.5
    assert repository.get("TR-42") == saved


def test_replaces_the_previous_prediction_for_the_same_transport_id():
    client = FakeSupabaseClient()
    repository = SupabasePredictionRepository(client)
    repository.save("TR-42", 20.0, "boxes", 1, "convnext_tiny")

    repository.save("TR-42", 75.0, "pallets", 2, "convnext_tiny")

    result = repository.get("TR-42")
    assert result is not None
    assert result.load_pct == 75.0
    assert result.models_count == 2


def test_returns_none_when_transport_id_is_not_saved():
    repository = SupabasePredictionRepository(FakeSupabaseClient())

    assert repository.get("TR-missing") is None

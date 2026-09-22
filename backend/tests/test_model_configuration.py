from app.estimator import DEFAULT_MODEL_PATH


def test_default_model_path_contains_predictor_checkpoints():
    """The service default must be usable without CV_MODEL_PATH."""
    assert DEFAULT_MODEL_PATH.is_dir()
    assert list(DEFAULT_MODEL_PATH.glob("fold*_best.pt"))

from src.config import settings


def test_request_timeout_has_a_positive_default():
    assert settings.request_timeout is not None
    assert settings.request_timeout > 0

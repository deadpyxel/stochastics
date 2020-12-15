import stochastics
from stochastics.config import VERSION


def test_returns_correct_version_number():
    assert stochastics.get_version() == VERSION

import pytest

from compose.metrics import across_seeds, continual_metrics, fewshot_summary


def test_continual_metrics():
    m = continual_metrics([[90.0], [80.0, 70.0], [60.0, 75.0, 50.0]], [90.0, 75.0, 60.0])
    assert m["Acon"] == 60.0
    assert m["AIA"] == pytest.approx(75.0)
    assert m["FF"] == pytest.approx(((90 - 60) + (75 - 75)) / 2)
    assert m["BWT"] == -m["FF"]


def test_fewshot_summary_without_sub_mode():
    s = fewshot_summary({"sys": [80.0, 80.0], "pro": [60.0, 60.0], "non": [70.0, 70.0], "noc": [90.0, 90.0]})
    assert s["Asys"] == [80.0, 0.0]
    assert s["Hn"] == pytest.approx(2 / (1 / 80 + 1 / 60))
    assert s["Ha"] == pytest.approx(4 / (1 / 80 + 1 / 60 + 1 / 70 + 1 / 90))


def test_across_seeds_uses_student_t():
    s = across_seeds([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["std"] == 1.0
    assert s["ci95"] == pytest.approx(4.303 / 3 ** 0.5)

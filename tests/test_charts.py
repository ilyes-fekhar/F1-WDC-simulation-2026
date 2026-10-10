import numpy as np
import pytest

pytest.importorskip("matplotlib")
import make_charts


def test_plot_clinch_writes_png(tmp_path):
    out = tmp_path / "c.png"
    make_charts.plot_clinch([0, 1, 27, 41, 22, 7, 2], list(range(17, 24)), {17: "Singapore"}, out,
                            "subtitle", "callout", "footer")
    assert out.stat().st_size > 5000


def test_surname():
    assert make_charts.surname("max_verstappen") == "Verstappen"
    assert make_charts.surname("antonelli") == "Antonelli"

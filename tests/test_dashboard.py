"""Headless dashboard tests using Streamlit's AppTest (runs app.py for real, no browser)."""
import pytest
from streamlit.testing.v1 import AppTest

import config

SECTIONS = ["System Overview", "Run Experiments", "Performance Comparison", "Machine Learning",
            "Execution Time Prediction", "Dataset & Results"]


def test_modules_import():
    import dashboard.components  # noqa: F401
    import dashboard.visualizations  # noqa: F401


@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_renders_without_exceptions(section):
    at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(section).run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.title[0].value == section.replace("&", "&")


def test_single_benchmark_button_runs_real_benchmark():
    at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("Run Experiments").run()
    at.number_input[0].set_value(20_000)
    at.button[0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Correctness validated" in s.value for s in at.success)

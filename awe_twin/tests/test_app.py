"""Headless app tests: every view, each main button, then a second render; synthetic and single uploaded unit;
HTML/CSS blocks must not print as text. Run: python -m pytest -q tests/test_app.py"""
import os
import runpy
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(HERE, "harness", "stub"), ROOT]
import streamlit as st  # noqa: E402  (stand-in)
import engine as E  # noqa: E402

VIEWS = ["Home", "Diagnostics", "Live twin", "Models", "Operations", "Study"]
FIRE = "Replay|Run assessment|Optimise|Update-frequency|Measurement ablation|no-leakage"


def render(view, fire=""):
    os.environ["VIEW"] = view
    st.FIRE = fire
    st.LOG.clear()
    for _ in range(4):
        try:
            runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
            break
        except st._Rerun:
            continue
        except st._StopRun:
            break
    return [m for k, m in st.LOG if k == "error"]


@pytest.mark.parametrize("view", VIEWS)
def test_synthetic_views(view):
    st.session_state.clear()
    st.session_state["syn_hours"] = 600
    st.session_state["m_units"] = ["S1", "S5"]
    st.session_state["m_models"] = ["Persistence", "Linear drift", "Bayesian ridge"]
    assert render(view, FIRE) == []
    assert render(view) == []            # second render with stored results


@pytest.mark.parametrize("view", VIEWS)
def test_single_uploaded_unit(view):
    st.session_state.clear()
    raw = pd.read_csv(os.path.join(ROOT, "sample_data", "example_single_cell_timeseries.csv"))
    df, _ = E.apply_mapping(raw, E.guess_mapping(list(raw.columns), "timeseries"), "timeseries", E.CellDesign(area_cm2=25))
    st.session_state["src"] = "My data (upload)"
    st.session_state["tables"] = {"timeseries": df}
    st.session_state["m_models"] = ["Persistence", "Linear drift"]
    assert render(view, FIRE) == []
    assert render(view) == []


def test_version_guard_strings():
    src = open(os.path.join(ROOT, "app.py")).read()
    assert f'REQUIRED_ENGINE = "{E.ENGINE_VERSION}"' in src

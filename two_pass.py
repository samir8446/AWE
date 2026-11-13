"""Press buttons, then render the page again with the stored results (what the user sees next).

Catches bugs that only appear on the re-render, e.g. `if not some_dataframe:` on a stored result.
Usage: APP=app.py VIEW="Models" FIRE="Train|Run" SETUP="Load demo" python two_pass.py
SETUP is pressed first (e.g. a demo-data button), then FIRE, then a plain re-render.
"""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.environ.get("APP", "app.py"))
sys.path[:0] = [os.path.join(HERE, "stub"), os.path.dirname(APP)]
os.chdir(os.path.dirname(APP))
import streamlit as st  # noqa: E402


def once(fire: str) -> None:
    st.FIRE = fire
    for _ in range(4):
        try:
            runpy.run_path(APP, run_name="__main__")
            return
        except st._Rerun:
            continue
        except st._StopRun:
            return


if os.environ.get("SETUP"):
    once(os.environ["SETUP"])
st.LOG.clear(); st.CHARTS.clear()
once(os.environ.get("FIRE", ""))
first = [m for k, m in st.LOG if k == "error"]
st.LOG.clear(); st.CHARTS.clear()
once("")
second = [m for k, m in st.LOG if k == "error"]
print("first run errors:", [m[:150] for m in first] or "none")
print("re-render:", "OK" if not second else [m[:150] for m in second])
print("charts:", [c for c in st.CHARTS if c])

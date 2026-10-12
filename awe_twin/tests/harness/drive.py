"""Run a Streamlit app headlessly through the stand-in in ./stub (no browser, no Streamlit install needed).

Usage:
  APP=path/to/app.py VIEW="Models" FIRE="Load demo|Train" SEL="Mode=measured;Models=2" python drive.py
  APP        path of the app (default ./app.py)
  VIEW       substring of the navigation option to open (segmented control / radio)
  FIRE       '|'-separated substrings: buttons whose label contains one of them return True
  SEL        ';'-separated label=value overrides for selectbox/radio/select_slider ('Models=2' = first 2 of a multiselect)
Prints errors/warnings shown by the app, the keys of the charts it drew, and the run time.
"""
import os, runpy, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.environ.get("APP", "app.py"))
sys.path[:0] = [os.path.join(HERE, "stub"), os.path.dirname(APP)]
os.chdir(os.path.dirname(APP))
import streamlit as st  # noqa: E402  (the stand-in)

t = time.time()
for _ in range(5):
    try:
        runpy.run_path(APP, run_name="__main__")
        break
    except st._Rerun:
        continue
    except st._StopRun:
        break
    except Exception:
        traceback.print_exc()
        break
print("issues:")
for kind, msg in st.LOG:
    if kind in ("error", "warning"):
        print(f"  {kind}: {msg[:200]}")
print("charts:", [c for c in st.CHARTS if c])
print(f"{time.time() - t:.1f}s")

"""Check that every HTML/<style> block the app emits through st.markdown(..., unsafe_allow_html=True) survives
CommonMark parsing (what Streamlit's Markdown follows). A <style> block ends at the line containing </style>: anything
appended after it on later lines is printed as visible text. Needs: pip install markdown-it-py.
Usage: APP=app.py SETUP="Load demo" python md_check.py
"""
import os, re, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.environ.get("APP", "app.py"))
sys.path[:0] = [os.path.join(HERE, "stub"), os.path.dirname(APP)]
os.chdir(os.path.dirname(APP))
import streamlit as st  # noqa: E402
from markdown_it import MarkdownIt  # noqa: E402

md = MarkdownIt("commonmark")
seen = []


def spy(body="", *a, **k):
    if k.get("unsafe_allow_html") and isinstance(body, str) and ("<style" in body or body.lstrip().startswith("<div")):
        seen.append(body)


st.markdown = spy
st._API["markdown"] = spy
for fire in (os.environ.get("SETUP", ""), ""):
    st.FIRE = fire
    for _ in range(3):
        try:
            runpy.run_path(APP, run_name="__main__")
            break
        except (st._Rerun, st._StopRun):
            continue
bad = [b for b in seen if "<style" in b and re.search(r"<p>[^<]*(\{|!important)", md.render(b))]
print(f"checked {len(seen)} HTML/style blocks, leaking as visible text: {len(bad)}")
for b in bad[:3]:
    print("  LEAK:", b[:100].replace("\n", " "))

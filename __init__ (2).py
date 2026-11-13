"""Minimal stand-in for Streamlit to run app.py headlessly (no browser). Widgets return defaults (or SEL overrides),
buttons return True when their label contains a FIRE substring; charts, errors and warnings are recorded."""
import os, types
__version__ = "1.46.0"
FIRE = os.environ.get("FIRE", "")
SEL = dict(x.rsplit("=", 1) for x in os.environ.get("SEL", "").split(";") if "=" in x)
LOG, CHARTS, UPLOADS = [], [], {}

class _SS(dict):
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v
    def __delattr__(self, k): self.pop(k, None)
session_state = _SS()

class _Any:
    """Absorbs any call/attribute; works as context manager and container."""
    def __getattr__(self, k):
        if k in _API: return _API[k]
        return _Any()
    def __call__(self, *a, **k): return _Any()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter([])
    def __bool__(self): return False

class _StopRun(Exception): pass
class _Rerun(Exception): pass

def _fmt(options, k):
    f = k.get("format_func")
    if f:
        for o in options: f(o)
def _fire(label): return any(f and f.lower() in str(label).lower() for f in FIRE.split("|"))
def _store(key, v):
    if key: session_state[key] = v
    return v
def button(label, *a, **k):
    v = _fire(label)
    if k.get("key"): session_state[k["key"]] = v     # like Streamlit: a keyed button owns its session key
    return v
def selectbox(label, options, index=0, *a, **k):
    options = list(options); _fmt(options, k)
    if not options: return None
    key = k.get("key")
    if key and key in session_state and session_state[key] in options: return session_state[key]
    v = SEL.get(label)
    if v is not None:
        for o in options:
            if v in str(o): return _store(key, o)
    i = k.get("index", index)
    return _store(key, options[i if isinstance(i, int) and i is not None and i < len(options) else 0])
def radio(label, options, index=0, *a, **k): return selectbox(label, options, index, **k)
def segmented_control(label, options, default=None, **k):
    options = list(options); _fmt(options, k)
    v = os.environ.get("VIEW")
    if v:
        for o in options:
            if v in str(o): return o
    return default if default is not None else options[0]
def select_slider(label, options=(), value=None, **k):
    options = list(options); _fmt(options, k)
    key = k.get("key")
    if key and key in session_state: return session_state[key]
    v = SEL.get(label)
    if v is not None:
        for o in options:
            if str(o) == v: return _store(key, o)
    return _store(key, value if value is not None else (options[0] if options else None))
def multiselect(label, options, default=None, **k):
    options = list(options); _fmt(options, k)
    key = k.get("key")
    if key and key in session_state: return [x for x in session_state[key] if x in options]
    v = SEL.get(label)
    if v is not None and v.isdigit(): return _store(key, options[:int(v)])
    return _store(key, [x for x in (default or []) if x in options])
def slider(label, min_value=None, max_value=None, value=None, *a, **k):
    key = k.get("key")
    if key and key in session_state: return session_state[key]
    return _store(key, value if value is not None else min_value)
def number_input(label, min_value=None, max_value=None, value=None, *a, **k):
    key = k.get("key")
    if key and key in session_state: return session_state[key]
    return _store(key, value if value is not None else (min_value if min_value is not None else 0))
def toggle(label, value=False, **k):
    v = SEL.get(label)
    return (v == "1") if v is not None else value
checkbox = toggle
def text_input(label, value="", **k): return value
def file_uploader(label, *a, **k):
    key = k.get("key")
    v = UPLOADS.get(key)
    if key: session_state[key] = v
    return v
def columns(spec, *a, **k):
    n = spec if isinstance(spec, int) else len(spec)
    return [_Any() for _ in range(n)]
def tabs(names): return [_Any() for _ in names]
def plotly_chart(fig, *a, **k): CHARTS.append(k.get("key")); return None
def error(msg, *a, **k): LOG.append(("error", str(msg)))
def warning(msg, *a, **k): LOG.append(("warning", str(msg)))
def info(msg, *a, **k): LOG.append(("info", str(msg)))
def exception(e, *a, **k): LOG.append(("error", repr(e)))
def stop(): raise _StopRun()
def rerun(): raise _Rerun()
def cache_data(f=None, **k):
    if f is None: return lambda g: _wrap(g)
    return _wrap(f)
def _wrap(g):
    g.clear = lambda *a, **k: None
    return g
cache_resource = cache_data
def fragment(f=None, **k): return f if f else (lambda g: g)
def progress(*a, **k): return _Any()
def empty(): return _Any()
def spinner(*a, **k): return _Any()
def expander(*a, **k): return _Any()
def container(*a, **k): return _Any()
def popover(*a, **k): return _Any()
def form(*a, **k): return _Any()
def status(*a, **k): return _Any()
def dialog(*a, **k): return lambda f: f
form_submit_button = button
sidebar = _Any()
secrets = {}
query_params = {}
context = types.SimpleNamespace(theme=types.SimpleNamespace(type="dark"))
def __getattr__(name): return _Any()
_API = {k: v for k, v in dict(globals()).items() if callable(v) and not k.startswith("__")}

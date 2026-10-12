from plotly import _Fig, _Obj
Figure = _Fig
def __getattr__(name):
    return lambda *a, **k: dict(k, _type=name)

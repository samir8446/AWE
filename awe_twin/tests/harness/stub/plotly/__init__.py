__version__ = "6.0.0"
class _Fig:
    """Absorbs any figure method; enough for the app to build charts headlessly."""
    def __init__(self, *a, **k): self.data = list(a[0]) if a and isinstance(a[0], (list, tuple)) else []
    def __getattr__(self, k):
        if k in ("layout",): return _Obj()
        return lambda *a, **kw: self
    def __iter__(self): return iter(self.data)
class _Obj:
    def __getattr__(self, k): return _Obj()
    def __call__(self, *a, **k): return _Obj()
    def __iter__(self): return iter([])
    def __bool__(self): return False

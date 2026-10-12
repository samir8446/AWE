def sample_colorscale(scale, samplepoints, *a, **k):
    n = samplepoints if isinstance(samplepoints, int) else len(list(samplepoints))
    return ["rgb(0,114,178)"] * max(n, 1)
def __getattr__(name): return lambda *a, **k: None

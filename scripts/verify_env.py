import importlib, traceback

mods = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("triton", "triton"),
    ("deepspeed", "deepspeed"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("cv2", "opencv-python-headless"),
    ("cupy", "cupy-cuda12x"),
    ("numba", "numba"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("datasets", "datasets"),
    ("ray", "ray"),
    ("vllm", "vllm"),
    ("verl", "verl"),
    ("lmms_eval", "lmms_eval"),
    ("fla", "flash-linear-attention"),
    ("flash_attn", "flash_attn"),
]

for mod, label in mods:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"OK    {label:26s} {v}")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:110]
        print(f"FAIL  {label:26s} {type(e).__name__}: {msg}")

import traceback

print("--- basic imports ---")
for mod in ["numpy", "pandas", "pyarrow", "datasets", "av", "torch", "deepspeed", "vllm", "ray", "verl", "lmms_eval"]:
    try:
        m = __import__(mod)
        print(f"OK    {mod:12s} {getattr(m,'__version__','?')}")
    except Exception as e:
        print(f"FAIL  {mod:12s} {type(e).__name__}: {str(e)[:100]}")

print("\n--- lmms_eval evaluator + task registry (eval pipeline) ---")
try:
    from lmms_eval import evaluator
    from lmms_eval.tasks import TaskManager
    tm = TaskManager(["vsibench", "cvbench", "blink_spatial", "sparbench"])
    print("OK    lmms_eval evaluator + TaskManager for all 4 benchmarks")
except Exception:
    print("FAIL  lmms_eval pipeline")
    traceback.print_exc()

print("\n--- datasets 5.x can read the cached CV-Bench parquet ---")
try:
    import datasets, os
    os.environ.setdefault("HF_HOME", "/home/c30084464/.cache/huggingface")
    d = datasets.load_dataset("nyu-visionx/CV-Bench", split="test")
    print(f"OK    CV-Bench loaded: {len(d)} rows, cols={d.column_names[:6]}")
except Exception:
    print("FAIL  CV-Bench load")
    traceback.print_exc()

print("\n--- training data pipeline: qwen_vl dataset module ---")
try:
    from qwen_vl.data import __init__ as _  # noqa
    print("OK    qwen_vl.data importable")
except Exception as e:
    try:
        import qwen_vl.data as qd
        print(f"OK    qwen_vl.data importable ({qd.__file__})")
    except Exception:
        print("FAIL  qwen_vl.data")
        traceback.print_exc()

import importlib.metadata as md
from packaging.requirements import Requirement
from packaging.version import Version

targets = {"numpy","protobuf","deepspeed","tenacity","datasets","sentencepiece","pyarrow","pandas","triton"}
rev = {t: [] for t in targets}

for dist in md.distributions():
    name = dist.metadata["Name"]
    reqs = dist.requires or []
    for r in reqs:
        try:
            req = Requirement(r)
        except Exception:
            continue
        base = req.name.lower().replace("_", "-")
        if base in targets:
            spec = str(req.specifier) if req.specifier else "(any)"
            marker = f"  [marker: {req.marker}]" if req.marker else ""
            rev[base].append((name, spec, marker))

def curver(t):
    try:
        return md.version(t)
    except Exception:
        return "(not installed)"

# desired downgrade targets from setup.py
spec_pin = {
    "numpy": "1.26.4",
    "protobuf": "3.20",
    "deepspeed": "0.16.4",
    "tenacity": "8.3.0",
    "datasets": "3.6.0",
    "sentencepiece": "0.1.99",
}

for t in sorted(targets):
    print(f"\n{'='*70}\n### {t}   current={curver(t)}   setup.py_pin={spec_pin.get(t,'(none)')}")
    if not rev[t]:
        print("   (no installed package declares a dependency on it)")
    for name, spec, marker in sorted(rev[t]):
        print(f"   - {name:32s} requires {t}{spec}{marker}")

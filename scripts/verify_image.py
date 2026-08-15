import traceback
import numpy as np

print("--- versions ---")
import cv2, PIL
from PIL import Image
print("cv2   ", cv2.__version__)
print("Pillow", PIL.__version__)
print("numpy ", np.__version__)

print("\n--- functional image ops (cv2) ---")
try:
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ok, buf = cv2.imencode(".jpg", img)
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    print(f"OK    resize{resized.shape} cvtColor{gray.shape} imencode/imdecode{dec.shape}")
except Exception:
    print("FAIL  cv2 ops"); traceback.print_exc()

print("\n--- functional image ops (PIL) ---")
try:
    im = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    im2 = im.resize((32, 32), Image.BICUBIC).convert("RGB")
    arr = np.asarray(im2)
    print(f"OK    PIL resize/convert -> {arr.shape} {arr.dtype}")
except Exception:
    print("FAIL  PIL ops"); traceback.print_exc()

print("\n--- read a real training frame (llava_hound jpeg) ---")
try:
    import glob
    files = glob.glob("data/media/llava_hound/frames/*/*.jpeg")[:1]
    if files:
        p = files[0]
        a = cv2.imread(p)
        b = np.asarray(Image.open(p).convert("RGB"))
        print(f"OK    cv2.imread{a.shape} PIL{b.shape}  ({p.split('/')[-2]})")
    else:
        print("SKIP  no llava_hound frames found")
except Exception:
    print("FAIL  real frame read"); traceback.print_exc()

print("\n--- vllm / verl stack still importable with opencv 4.11 ---")
for mod in ["vllm", "ray", "verl", "lmms_eval", "transformers", "deepspeed"]:
    try:
        m = __import__(mod)
        print(f"OK    {mod:12s} {getattr(m,'__version__','?')}")
    except Exception as e:
        print(f"FAIL  {mod:12s} {type(e).__name__}: {str(e)[:100]}")

print("\n--- vllm multimodal image path (uses cv2) ---")
try:
    from vllm.multimodal import MULTIMODAL_REGISTRY  # noqa
    print("OK    vllm.multimodal registry imported")
except Exception as e:
    print(f"WARN  vllm.multimodal: {type(e).__name__}: {str(e)[:140]}")

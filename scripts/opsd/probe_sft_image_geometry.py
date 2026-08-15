"""Report the exact per-view geometry SFT feeds the model, via prepare_image_inputs.

Also reports what an unconstrained load would cost, which is what verl's teacher
branch does today (raw Image.open, no resize).
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

import transformers
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from qwen_vl.data.utils import load_and_preprocess_images, prepare_image_inputs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="output/spatialstack_qwen35_novggt_aligned")
    ap.add_argument("--annotation", default="data/annotations/spar_234k.json")
    ap.add_argument("--media_root", default="data/media")
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    processor = transformers.AutoProcessor.from_pretrained(args.model_path)
    ip = processor.image_processor
    ip.max_pixels = 576 * 28 * 28
    ip.min_pixels = 16 * 28 * 28
    ip.size["longest_edge"] = ip.max_pixels
    ip.size["shortest_edge"] = ip.min_pixels
    merge = ip.merge_size
    print(f"patch_size={ip.patch_size} merge_size={merge} factor={ip.patch_size * merge}")

    annotations = json.load(open(args.annotation))
    random.seed(0)
    stats = Counter()
    unconstrained = []
    for n_views in (3, 32):
        pool = [a for a in annotations if len(a.get("images", [])) == n_views]
        for ann in random.sample(pool, args.samples):
            path = os.path.join(args.media_root, ann["images"][0])
            vggt = load_and_preprocess_images([path])
            ret = prepare_image_inputs(path, ip, model_type="qwen3.5")
            t, gh, gw = ret["image_grid_thw"].tolist()
            tokens = t * gh * gw // (merge**2)
            stats[(n_views, tuple(vggt.shape[-2:]), (gh, gw), tokens)] += 1

            w, h = Image.open(path).size
            factor = ip.patch_size * merge
            unconstrained.append((w // factor) * (h // factor))

    print(f"\n{'views':>6} {'VGGT HxW':>12} {'grid(h,w)':>12} {'tokens/view':>12} {'count':>6}")
    print("-" * 54)
    for (nv, hw, grid, tok), c in sorted(stats.items()):
        print(f"{nv:>6} {str(hw):>12} {str(grid):>12} {tok:>12} {c:>6}")

    mean_raw = sum(unconstrained) / len(unconstrained)
    print(f"\nunconstrained (verl teacher path today): {mean_raw:.0f} tok/view mean, "
          f"{min(unconstrained)}-{max(unconstrained)} range")

    per_view = sorted({k[3] for k in stats})
    print(f"\ntokens/view observed: {per_view}")
    for tok in per_view:
        for n in (8, 16, 32):
            print(f"  {tok} tok/view x {n:>2} views = {tok * n:>6} visual tokens")


if __name__ == "__main__":
    main()

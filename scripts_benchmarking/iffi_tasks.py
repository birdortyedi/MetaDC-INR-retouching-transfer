#!/usr/bin/env python
"""Task definition for the zero-shot IFFI evaluation (Table 6), read in place from the IFFI test split.

  * layout: <root>/Original/<img> (unedited) and <root>/<filter>/<img> (filtered), 16 filters;
  * images sorted numerically; the reference pair of each target is the next image (cyclic);
  * every image is decoded with PIL and capped at 768 px on the longest side (bilinear);
  * the 12 images whose filtered versions differ in size from the original are excluded, and the
    reference is the next kept image. This gives 88 targets x 16 filters = 1,408 tasks.

    python scripts_benchmarking/iffi_tasks.py --iffi_root <IFFI>/test        prints the task count
"""
import argparse, os
import numpy as np
from PIL import Image

IFFI_ROOT = os.environ.get("IFFI_ROOT")   # or pass root=... / --iffi_root
MAX_SIZE = 768
EXPECT_EXCLUDED = 12


def numeric_key(x):
    return int("".join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x


def capped_pil(path, max_size=MAX_SIZE):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max_size > 0 and max(w, h) > max_size:
        s = max_size / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.Resampling.BILINEAR)
    return img


def filters(root=IFFI_ROOT):
    return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d)) and d.lower() != "original"]


def kept_images(root=IFFI_ROOT):
    orig = os.path.join(root, "Original")
    imgs = sorted([f for f in os.listdir(orig) if f.endswith((".jpg", ".png", ".jpeg"))], key=numeric_key)
    fs = filters(root)
    bad = []
    for im in imgs:
        s0 = Image.open(os.path.join(orig, im)).size
        if any(not os.path.exists(os.path.join(root, f, im)) or Image.open(os.path.join(root, f, im)).size != s0
               for f in fs):
            bad.append(im)
    assert len(bad) == EXPECT_EXCLUDED, f"expected {EXPECT_EXCLUDED} excluded images, found {len(bad)}: {bad}"
    return [im for im in imgs if im not in bad], bad


def tasks(root=IFFI_ROOT):
    kept, _ = kept_images(root)
    out = []
    for i, tgt in enumerate(kept):
        ref = kept[(i + 1) % len(kept)]
        for f in filters(root):
            out.append(dict(filter=f, target=tgt, reference=ref,
                            t_nat=os.path.join(root, "Original", tgt), r_nat=os.path.join(root, "Original", ref),
                            r_ret=os.path.join(root, f, ref), gt=os.path.join(root, f, tgt)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iffi_root", default=IFFI_ROOT, required=IFFI_ROOT is None)
    ap.add_argument("--write_refs", default=None, help="optional: write 'target,reference' lines")
    a = ap.parse_args()
    kept, bad = kept_images(a.iffi_root)
    t = tasks(a.iffi_root)
    print(f"filters={len(filters(a.iffi_root))} kept={len(kept)} excluded={bad} tasks={len(t)}")
    if a.write_refs:
        with open(a.write_refs, "w") as fh:
            for i, im in enumerate(kept):
                fh.write(f"{im},{kept[(i + 1) % len(kept)]}\n")
        print("wrote", a.write_refs)

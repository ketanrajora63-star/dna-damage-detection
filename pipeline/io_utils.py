import os, glob, json
from typing import List, Tuple, Optional
import numpy as np
from skimage import io as skio

def list_images_recursive(input_dir: str, image_exts: List[str]) -> List[str]:
    paths = []
    for ext in image_exts:
        paths.extend(glob.glob(os.path.join(input_dir, f"**/*{ext}"), recursive=True))
    return sorted(list(dict.fromkeys(paths)))

def load_split_channels(path: str, r_suffix: str, g_suffix: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    base = path
    ext = os.path.splitext(base)[1]
    if base.endswith(r_suffix + ext):
        base = base[: -len(r_suffix + ext)]
    elif base.endswith(g_suffix + ext):
        base = base[: -len(g_suffix + ext)]
    r_path = base + r_suffix + ext
    g_path = base + g_suffix + ext

    R = skio.imread(r_path)
    G = skio.imread(g_path) if os.path.exists(g_path) else None
    if R.ndim > 2:
        if R.shape[-1] in (3,4):
            R = R[...,0]
        else:
            R = R[0]
    if G is not None and G.ndim > 2:
        if G.shape[-1] in (3,4):
            G = G[...,1]
        else:
            G = G[0]
    return R, G

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = np.min(img), np.max(img)
    if mx > mn:
        img = (img - mn) / (mx - mn)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img

from typing import Dict, List
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor, laplace
from skimage.measure import regionprops, moments_hu

def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn)/(mx - mn)
    return (img*255).clip(0,255).astype(np.uint8)

def shape_features(mask: np.ndarray) -> Dict[str, float]:
    props = regionprops(mask.astype(int))
    if not props:
        return {"area":0,"major_axis":0,"minor_axis":0,"axis_ratio":0,"eccentricity":0,"solidity":0,"roundness":0,"extent":0}
    p = props[0]
    roundness = 4*np.pi*p.area/((p.perimeter**2)+1e-6) if p.perimeter>0 else 0
    axis_ratio = (p.major_axis_length/(p.minor_axis_length+1e-6)) if p.minor_axis_length>0 else 0
    return {
        "area": float(p.area),
        "major_axis": float(p.major_axis_length),
        "minor_axis": float(p.minor_axis_length),
        "axis_ratio": float(axis_ratio),
        "eccentricity": float(p.eccentricity),
        "solidity": float(getattr(p, "solidity", 0.0)),
        "roundness": float(roundness),
        "extent": float(p.extent),
    }

def haralick_features(img: np.ndarray, distances: List[int], angles: List[float]) -> Dict[str, float]:
    I = _to_uint8(img)
    glcm = greycomatrix(I, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    feats = {}
    for prop in ["contrast","dissimilarity","homogeneity","ASM","energy","correlation"]:
        try:
            vals = greycoprops(glcm, prop)
            for j, d in enumerate(distances):
                for i in range(len(angles)):
                    feats[f"har_{prop}_d{d}_a{i}"] = float(vals[j, i])
        except Exception:
            pass
    return feats

def lbp_features(img: np.ndarray, radius: int, n_points: int) -> Dict[str, float]:
    I = _to_uint8(img)
    lbp = local_binary_pattern(I, P=n_points, R=radius, method="uniform").astype(np.int32)
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return {f"lbp_{i}": float(v) for i, v in enumerate(hist)}

def hog_features(img: np.ndarray, pixels_per_cell: int, cells_per_block: int) -> Dict[str, float]:
    I = _to_uint8(img)
    vec = hog(I, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
              cells_per_block=(cells_per_block, cells_per_block),
              block_norm="L2-Hys", visualize=False, feature_vector=True)
    return {f"hog_{i}": float(v) for i, v in enumerate(vec.ravel())}

def gabor_features(img: np.ndarray, frequencies: List[float], thetas: List[float]) -> Dict[str, float]:
    I = _to_uint8(img)
    feats = {}; idx = 0
    from skimage.filters import gabor
    for f in frequencies:
        for t in thetas:
            real, imag = gabor(I, frequency=f, theta=t)
            feats[f"gabor_{idx}_real_mean"] = float(real.mean())
            feats[f"gabor_{idx}_real_std"]  = float(real.std())
            feats[f"gabor_{idx}_imag_mean"] = float(imag.mean())
            feats[f"gabor_{idx}_imag_std"]  = float(imag.std())
            idx += 1
    return feats

def log_features(img: np.ndarray) -> Dict[str, float]:
    I = _to_uint8(img).astype(np.float32)
    L = laplace(I)
    return {"log_mean": float(L.mean()), "log_std": float(L.std()), "log_energy": float((L**2).mean())}

def hu_moments_features(img: np.ndarray) -> Dict[str, float]:
    I = _to_uint8(img)
    m = cv2.moments(I)
    hu = moments_hu(m)
    return {f"hu_{i}": float(v) for i, v in enumerate(hu.ravel())}

def extract_all_features(crop, mask, distances, angles, lbp_radius, lbp_n_points,
                         hog_pixels_per_cell, hog_cells_per_block,
                         gabor_frequencies, gabor_thetas):
    feats = {}
    feats.update(shape_features(mask))
    feats.update(haralick_features(crop, distances, angles))
    feats.update(lbp_features(crop, lbp_radius, lbp_n_points))
    feats.update(hog_features(crop, hog_pixels_per_cell, hog_cells_per_block))
    feats.update(gabor_features(crop, gabor_frequencies, gabor_thetas))
    feats.update(log_features(crop))
    feats.update(hu_moments_features(crop))
    return feats

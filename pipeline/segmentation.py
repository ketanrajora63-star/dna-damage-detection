from typing import Tuple, Dict, Any
import numpy as np
import cv2
from skimage import filters, morphology, measure, segmentation, exposure

def segment_nuclei_otsu(r_img: np.ndarray, gaussian_sigma: float = 1.0, min_area: int = 150, max_area: int = 999999):
    r = cv2.GaussianBlur(r_img, (0,0), gaussian_sigma)
    p2, p98 = np.percentile(r, (2, 98))
    r_cs = exposure.rescale_intensity(r, in_range=(p2, p98))
    thr = filters.threshold_otsu(r_cs)
    bw = (r_cs > thr).astype(np.uint8)
    bw = morphology.remove_small_objects(bw.astype(bool), min_size=min_area)
    bw = morphology.remove_small_holes(bw, area_threshold=64)
    bw = bw.astype(np.uint8)
    distance = cv2.distanceTransform((bw*255).astype(np.uint8), cv2.DIST_L2, 3)
    local_max = filters.rank.maximum((distance/distance.max()*255).astype(np.uint8), morphology.disk(3))
    markers = measure.label(local_max > np.percentile(local_max, 95))
    labels = segmentation.watershed(-distance, markers, mask=bw)
    props = measure.regionprops(labels)
    keep = np.zeros_like(labels)
    idx = 1
    kept = []
    for p in props:
        if min_area <= p.area <= max_area:
            keep[labels == p.label] = idx
            kept.append(p)
            idx += 1
    return keep, kept

def segment_nuclei(r_img: np.ndarray, method: str = "otsu", **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    if method == "otsu":
        labels, props = segment_nuclei_otsu(r_img, **kwargs)
        return labels, {"props": props, "method": "otsu"}
    raise NotImplementedError(f"Segmentation method '{method}' not implemented yet.")

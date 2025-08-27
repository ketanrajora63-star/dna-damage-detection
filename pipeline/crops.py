from typing import Tuple
import numpy as np
import cv2
from skimage import measure

def _rotate_to_major_axis(image: np.ndarray, mask: np.ndarray):
    props = measure.regionprops(mask.astype(int))
    if not props:
        return image, mask
    p = props[0]
    angle_deg = -(p.orientation * 180.0 / np.pi)
    rot = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle_deg, 1.0)
    img_rot = cv2.warpAffine(image, rot, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    msk_rot = cv2.warpAffine((mask>0).astype(np.uint8)*255, rot, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_rot, (msk_rot>0).astype(np.uint8)

def extract_nucleus_crop(image: np.ndarray, labels: np.ndarray, label_id: int, crop_size: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    m = (labels == label_id).astype(np.uint8)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None, None
    x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
    pad = 10
    x0 = max(0, x0 - pad); x1 = min(image.shape[1]-1, x1 + pad)
    y0 = max(0, y0 - pad); y1 = min(image.shape[0]-1, y1 + pad)

    roi = image[y0:y1+1, x0:x1+1]
    mroi = m[y0:y1+1, x0:x1+1]
    roi, mroi = _rotate_to_major_axis(roi, mroi)

    h, w = roi.shape[:2]; side = max(h, w)
    pad_top = (side - h)//2; pad_bottom = side - h - pad_top
    pad_left= (side - w)//2; pad_right  = side - w - pad_left
    roi = cv2.copyMakeBorder(roi, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    mroi= cv2.copyMakeBorder(mroi, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    roi = cv2.resize(roi, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    mroi= cv2.resize(mroi, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    return roi, (mroi>0).astype(np.uint8)

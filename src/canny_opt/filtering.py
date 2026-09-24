import cv2
import numpy as np
from scipy import ndimage as ndi


def adaptive_median_filter(img, s_init=3, s_max=7, on_noise="median"):

    if img.ndim != 2:
        raise ValueError("A imagem precisa estar em tons de cinza (2D).")
    if s_init % 2 == 0 or s_max % 2 == 0 or s_max < s_init:
        raise ValueError("s_init e s_max devem ser ímpares e s_max >= s_init.")
    if on_noise not in ("median", "min"):
        raise ValueError("on_noise deve ser 'median' ou 'min'.")

    img = img.astype(np.int16)
    out = img.copy()
    done = np.zeros(img.shape, dtype=bool)

    for s in range(s_init, s_max + 1, 2):
        z_min = ndi.minimum_filter(img, size=s, mode="reflect")
        z_max = ndi.maximum_filter(img, size=s, mode="reflect")
        z_mid = ndi.median_filter(img, size=s, mode="reflect")

        a_ok = ((z_mid - z_min) > 0) & ((z_mid - z_max) < 0)
        go_b = a_ok & ~done

        b_ok = ((img - z_min) > 0) & ((img - z_max) < 0)
        z_noise = z_mid if on_noise == "median" else z_min
        out[go_b] = np.where(b_ok, img, z_noise)[go_b]
        done |= go_b

    return np.clip(out, 0, 255).astype(np.uint8)


def morphological_closing(img, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def adaptive_smooth_filter(img, s_init=3, s_max=7, close_ksize=3, on_noise="median"):
    filtered = adaptive_median_filter(img, s_init, s_max, on_noise)
    return morphological_closing(filtered, close_ksize)
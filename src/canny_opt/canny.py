# canny.py
import numpy as np
import cv2
from .filtering import adaptive_smooth_filter
from .gradient_and_threshold import (
    calculate_8direction_gradient, calculate_automatic_thresholds,
)

def non_maximum_suppression(mag, angle):
    ang = np.rad2deg(angle) % 180
    def at(dy, dx):  # valor em (i+dy, j+dx)
        return np.roll(np.roll(mag, -dy, axis=0), -dx, axis=1)

    b0 = (ang < 22.5) | (ang >= 157.5)
    b1 = (ang >= 22.5) & (ang < 67.5)
    b2 = (ang >= 67.5) & (ang < 112.5)
    # b3: restante (112.5–157.5)

    p1 = np.where(b0, at(0, 1),  np.where(b1, at(1, 1),  np.where(b2, at(1, 0), at(1, -1))))
    p2 = np.where(b0, at(0, -1), np.where(b1, at(-1, -1), np.where(b2, at(-1, 0), at(-1, 1))))

    nms = np.where((mag >= p1) & (mag >= p2), mag, 0).astype(np.float32)
    nms[[0, -1], :] = 0
    nms[:, [0, -1]] = 0
    return nms

def hysteresis_thresholding(nms, T_low, T_high):
    strong = nms >= T_high
    cand = (nms >= T_low).astype(np.uint8)
    _, lab = cv2.connectedComponents(cand, connectivity=8)
    keep = np.unique(lab[strong])
    keep = keep[keep > 0]
    return (np.isin(lab, keep) * 255).astype(np.uint8)

def improved_canny(img):
    filtered = adaptive_smooth_filter(img)
    mag, angle, _ = calculate_8direction_gradient(filtered)
    mag = mag / (mag.max() + 1e-8) * 255.0               # normaliza p/ 0–255
    nms = non_maximum_suppression(mag, angle)
    T_high, T_low, _, _ = calculate_automatic_thresholds(nms)
    return hysteresis_thresholding(nms, T_low, T_high)
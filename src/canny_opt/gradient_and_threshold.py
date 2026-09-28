# gradient_and_threshold.py
import numpy as np
import cv2

def get_8direction_templates(size=5, sigma_d=1.0, sigma_s=1.5):
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    kernels = []
    for k in range(8):
        th = np.deg2rad(22.5 * k)
        u = x * np.cos(th) + y * np.sin(th)      # direção do gradiente
        v = -x * np.sin(th) + y * np.cos(th)     # direção da borda
        K = u * np.exp(-u**2 / (2 * sigma_d**2)) * np.exp(-v**2 / (2 * sigma_s**2))
        K -= K.mean()                             # garante soma zero
        K /= np.abs(K).sum()
        kernels.append(K.astype(np.float32))
    return kernels

_TEMPLATES = get_8direction_templates()

def calculate_8direction_gradient(img):
    img_f = img.astype(np.float32)
    responses = [cv2.filter2D(img_f, cv2.CV_32F, K) for K in _TEMPLATES]
    S = np.sqrt(sum(r**2 for r in responses))            # Eq. (5)
    A = np.arctan2(responses[4], responses[0])           # S90 = gy, S0 = gx
    return S, A, responses

def adaptive_iterative_threshold(vals, C=0.5, max_iter=100):
    T = 0.5 * (vals.max() + vals.min())
    for _ in range(max_iter):
        lo, hi = vals[vals < T], vals[vals >= T]
        T1 = lo.mean() if lo.size else 0.0
        T2 = hi.mean() if hi.size else 0.0
        T_new = 0.5 * (T1 + T2)
        if abs(T_new - T) < C:
            return float(0.5 * (T_new + T))
        T = T_new
    return float(T)

def otsu_threshold(vals):
    v = np.clip(vals, 0, 255).astype(np.uint8).reshape(-1, 1)
    T, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(T)

def calculate_automatic_thresholds(nms):
    vals = nms[nms > 0].astype(np.float32)               # só pixels pós-NMS
    T_L1 = adaptive_iterative_threshold(vals)
    T_L2 = otsu_threshold(vals)
    T_high = 0.5 * (T_L1 + T_L2)                         # Eq. (14)
    T_low = 0.5 * T_high                                 # Eq. (15)
    return T_high, T_low, T_L1, T_L2
import numpy as np
import cv2

# =====================================================================
# SEÇÃO II.B - GRADIENTE EM 8 DIREÇÕES (Eq. 4 e 5)
# =====================================================================

# Matrizes da Eq. (4), transcritas do PDF exatamente como impressas.
PAPER_TEMPLATES = {
    0.0:   [[ 0,  0,  0,  0,  0], [-1, -2, -4, -2, -1], [ 0,  0,  0,  0,  0], [ 1,  2,  4,  2,  1], [ 0,  0,  0,  0,  0]],
    22.5:  [[ 0,  0,  0,  0,  0], [ 0, -2, -4, -2,  0], [ 0, -4,  0,  4,  0], [ 0,  2,  4,  2,  0], [ 0,  0,  0,  0,  0]],
    45.0:  [[ 0,  0,  0, -1,  0], [ 0, -2, -4,  0,  1], [ 0, -4,  0,  4,  0], [-1,  0,  4,  2,  0], [ 0,  1,  0,  0,  0]],
    67.5:  [[ 0,  0, -1,  0,  0], [ 0, -2, -4,  2,  0], [ 0, -4,  0,  4,  0], [ 0,  2,  4,  2,  0], [ 0,  0,  1,  0,  0]],
    90.0:  [[ 0, -1,  0,  1,  0], [ 0, -2,  0,  2,  0], [ 0, -4,  0,  4,  0], [ 0, -2,  0,  2,  0], [ 0, -1,  0,  1,  0]],
    112.5: [[ 0,  0,  1,  0,  0], [ 0, -2, -4,  2,  0], [ 0, -4,  0,  4,  0], [ 0,  2, -4,  2,  0], [ 0,  0, -1,  0,  0]],
    135.0: [[ 0,  1,  0,  0,  0], [-1,  0,  4,  2,  0], [ 0, -4,  0,  4,  0], [ 0, -2, -4,  0,  1], [ 0,  0,  0, -1,  0]],
    157.5: [[ 0,  0,  0,  0,  0], [ 0,  2,  4,  2,  0], [-1, -4,  0,  4,  1], [ 0, -2, -4, -2,  0], [ 0,  0,  0,  0,  0]],
}


def get_8direction_templates(fix_zero_sum=True):
    
    kernels = []
    for angle in sorted(PAPER_TEMPLATES):
        K = np.array(PAPER_TEMPLATES[angle], dtype=np.float32)
        s = K.sum()
        if fix_zero_sum and s != 0:
            nz = K != 0
            K[nz] -= s / nz.sum()
        kernels.append(K)
    return kernels


_TEMPLATES = get_8direction_templates()


def calculate_8direction_gradient(img):
    
    img_f = img.astype(np.float32)
    responses = [cv2.filter2D(img_f, cv2.CV_32F, K) for K in _TEMPLATES]

    S = np.sqrt(sum(r ** 2 for r in responses))
    S0, S90 = responses[0], responses[4]
    A = np.arctan2(S0, S90)
    return S, A, responses


# =====================================================================
# SEÇÃO II.C - LIMIARES AUTOMÁTICOS (Eq. 6 a 15)
# =====================================================================

def adaptive_iterative_threshold(vals, C=0.5, max_iter=100):
    """T_L1 pelo método iterativo (Eq. 6 a 11)."""
    T = 0.5 * (vals.max() + vals.min())                      # Eq. (6)
    for _ in range(max_iter):
        lo, hi = vals[vals < T], vals[vals >= T]
        T1 = lo.mean() if lo.size else 0.0                   # Eq. (7)
        T2 = hi.mean() if hi.size else 0.0                   # Eq. (8)
        T_new = 0.5 * (T1 + T2)                              # Eq. (9)
        T_L = 0.5 * (T_new + T)                              # Eq. (11)
        if abs(T_new - T) < C:                               # Eq. (10)
            return float(T_L)
        T = T_L                                              # novo limiar = média
    return float(T)


def otsu_threshold(vals):
    """T_L2 por Otsu (Eq. 12 e 13), sobre 8 bits."""
    v = np.clip(vals, 0, 255).astype(np.uint8).reshape(-1, 1)
    T, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(T)


def calculate_automatic_thresholds(nms):
    """Eq. (14) e (15). Usa só os pixels que sobraram da NMS (> 0)."""
    vals = nms[nms > 0].astype(np.float32)
    T_L1 = adaptive_iterative_threshold(vals)
    T_L2 = otsu_threshold(vals)
    T_high = 0.5 * (T_L1 + T_L2)
    T_low = 0.5 * T_high
    return T_high, T_low, T_L1, T_L2
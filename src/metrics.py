import numpy as np
from skimage.metrics import structural_similarity as ssim_func

def mse(img1, img2):
    """Mean Squared Error (Erro Quadrático Médio)"""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))

def psnr(img1, img2):
    """Peak Signal-to-Noise Ratio (Relação Sinal-Ruído de Pico)"""
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    max_pixel = 255.0
    return float(20 * np.log10(max_pixel / np.sqrt(mse_val)))

def ssim(img1, img2):
    """Structural Similarity Index Metric (Índice de Similaridade Estrutural)"""
    return float(ssim_func(img1, img2, data_range=255))
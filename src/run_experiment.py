import argparse
from pathlib import Path
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from filtering import adaptive_smooth_filter
from metrics import mse, psnr, ssim
from noise import add_salt_and_pepper

ROOT = Path(__file__).resolve().parent.parent

PIPELINES = {
    "Gaussiano 5x5 (Canny)": lambda img: cv2.GaussianBlur(img, (5, 5), 1.0),
    "Mediana 3x3": lambda img: cv2.medianBlur(img, 3),
    "Adaptativo + closing (II.A)": adaptive_smooth_filter,
}


def load_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in exts)
    images = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths}
    if not images:
        print(f"[aviso] nenhuma imagem em '{folder}'; usando 'cameraman' do scikit-image.")
        from skimage import data
        images = {"cameraman": data.camera()}
    return images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "images"))
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.15, 0.30])
    args = ap.parse_args()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"{'Imagem':<12} {'Ruído':>6} | {'Método':<28} | {'MSE':>9} {'PSNR':>7} {'SSIM':>6}")
    print("-" * 80)

    for name, original in load_images(args.images).items():
        ncols = len(PIPELINES) + 1
        fig, axes = plt.subplots(len(args.noise), ncols,
                                 figsize=(4 * ncols, 4 * len(args.noise)),
                                 squeeze=False)
        for row, density in enumerate(args.noise):
            noisy = add_salt_and_pepper(original, density)
            outputs = {"Ruidosa": noisy}
            outputs.update({k: f(noisy) for k, f in PIPELINES.items()})

            for col, (method, out) in enumerate(outputs.items()):
                print(f"{name:<12} {density:>6.0%} | {method:<28} | "
                      f"{mse(original, out):>9.2f} {psnr(original, out):>7.2f} {ssim(original, out):>6.3f}")
                axes[row, col].imshow(out, cmap="gray", vmin=0, vmax=255)
                axes[row, col].set_title(f"{method}\nruído {density:.0%}", fontsize=9)
                axes[row, col].axis("off")
            print("-" * 80)

        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_filtragem.png", dpi=110)
        plt.close(fig)

    print(f"Figuras salvas em {out_dir}")


if __name__ == "__main__":
    main()
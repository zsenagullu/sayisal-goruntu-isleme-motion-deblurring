"""Richardson-Lucy ile hareket bulanıklığı giderme projesi.

Kullanım:
  python main.py --demo --blur-lengths 9 --angles 45 --iterations 30 --max-side 512
  python main.py --blur-lengths 9 17 25 --angles 0 45 90 --iterations 30

Net fotoğraflar data/raw klasörüne konur. Program bu fotoğraflardan kontrollü
hareket bulanıklığı üretir; gerçek bulanık fotoğraflar için ayrı bir akış
sonradan eklenecektir.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import fftconvolve
from skimage import data
from skimage.draw import line
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import richardson_lucy

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs" / "comparisons"
RESTORED_DIR = BASE_DIR / "outputs" / "restored"
RESULTS_DIR = BASE_DIR / "results"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resize_rgb(image: np.ndarray, max_side: int) -> np.ndarray:
    """RGB görüntüyü en uzun kenarı max_side olacak şekilde küçültür; büyütmez."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Görüntü RGB biçiminde olmalı.")
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / longest
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        pil_image = Image.fromarray(image)
        image = np.asarray(pil_image.resize(new_size, Image.Resampling.LANCZOS))
    return image.astype(np.float32) / 255.0


def load_rgb_image(path: Path, max_side: int) -> np.ndarray:
    """Görüntüyü RGB olarak yükler ve bellekteki boyutunu sınırlar."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        array = np.asarray(image)
    return resize_rgb(array, max_side)


def make_motion_psf(length: int, angle_degrees: float) -> np.ndarray:
    """Verilen uzunluk/yönde, toplamı 1 olan doğrusal hareket bulanıklığı çekirdeği."""
    if length < 3 or length % 2 == 0:
        raise ValueError("Bulanıklık uzunluğu 3 veya daha büyük tek sayı olmalı (örn. 9, 17, 25).")
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    half = center
    angle = np.deg2rad(angle_degrees)
    x_start = int(round(center - half * np.cos(angle)))
    y_start = int(round(center - half * np.sin(angle)))
    x_end = int(round(center + half * np.cos(angle)))
    y_end = int(round(center + half * np.sin(angle)))
    rows, cols = line(y_start, x_start, y_end, x_end)
    kernel[rows, cols] = 1.0
    total = kernel.sum()
    if total == 0:
        raise RuntimeError("Hareket bulanıklığı çekirdeği oluşturulamadı.")
    return kernel / total


def convolve_rgb(rgb: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Hareket PSF'sini RGB kanallarına FFT tabanlı evrişimle uygular."""
    result = np.empty_like(rgb, dtype=np.float32)
    for channel in range(3):
        result[:, :, channel] = fftconvolve(rgb[:, :, channel], psf, mode="same")
    # FFT yuvarlama nedeniyle oluşabilecek çok küçük negatif/aşırı değerleri sınırla.
    return np.clip(result, 0.0, 1.0)


def add_motion_blur_with_padding(
    rgb: np.ndarray, psf: np.ndarray, length: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Yansımalı kenar genişletmesiyle bulanıklık üretir.

    Önceki demoda kenar payı yalnızca PSF uzunluğu kadardı. RL yinelemeleri
    sırasında sınır etkilerinin kırpılan alana taşınmasını azaltmak için payı
    en az iki PSF uzunluğuna çıkarıyoruz. Değerler (bulanık genişletilmiş,
    kırpılmış bulanık görüntü, pad) olarak döner.
    """
    height, width = rgb.shape[:2]
    pad = max(2 * length, 32)
    padded = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    blurred_padded = convolve_rgb(padded, psf)
    blurred_crop = blurred_padded[pad:pad + height, pad:pad + width]
    return blurred_padded, blurred_crop, pad


def restore_rgb(blurred: np.ndarray, psf: np.ndarray, iterations: int) -> np.ndarray:
    """Richardson-Lucy algoritmasını her RGB kanalına ayrı uygular."""
    restored = np.empty_like(blurred, dtype=np.float32)
    for channel in range(3):
        restored[:, :, channel] = richardson_lucy(
            blurred[:, :, channel], psf, num_iter=iterations, clip=False
        )
    return np.clip(restored, 0.0, 1.0)


def tv_richardson_lucy(
    image: np.ndarray,
    psf: np.ndarray,
    num_iter: int = 30,
    lmbda: float = 0.0012,
) -> np.ndarray:
    """Total Variation (TV) düzenlileştirmeli Richardson-Lucy algoritması (Dey et al., 2006).

    Standart Richardson-Lucy güncellemesini TV diverjans terimiyle sönümler:
    u^(t+1) = [u^(t) * ( (g / (h * u^(t))) * h* )] / [1 - lambda * div(grad u / |grad u|)]
    """
    float_type = np.float32
    u = np.full(image.shape, 0.5, dtype=float_type)
    psf_mirror = np.flip(psf)
    eps = 1e-12
    eps_tv = 1e-3

    for _ in range(num_iter):
        conv = fftconvolve(u, psf, mode="same") + eps
        ratio = image / conv
        factor = fftconvolve(ratio, psf_mirror, mode="same")

        if lmbda > 0:
            gy, gx = np.gradient(u)
            norm = np.sqrt(gx**2 + gy**2 + eps_tv**2)
            nx = gx / norm
            ny = gy / norm
            _, dnx_dx = np.gradient(nx)
            dny_dy, _ = np.gradient(ny)
            div = dnx_dx + dny_dy
            denom = np.clip(1.0 - lmbda * div, 0.1, 10.0)
            u = (u * factor) / denom
        else:
            u = u * factor

        u = np.clip(u, 0.0, 1.0)

    return u


def restore_rgb_tv(
    blurred: np.ndarray, psf: np.ndarray, iterations: int, tv_weight: float
) -> np.ndarray:
    """TV düzenlileştirmeli Richardson-Lucy algoritmasını her RGB kanalına uygular."""
    restored = np.empty_like(blurred, dtype=np.float32)
    for channel in range(3):
        restored[:, :, channel] = tv_richardson_lucy(
            blurred[:, :, channel], psf, num_iter=iterations, lmbda=tv_weight
        )
    return np.clip(restored, 0.0, 1.0)


def save_rgb(array: np.ndarray, path: Path) -> None:
    """Sonucu orijinal deney çözünürlüğünde PNG olarak kaydeder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(output).save(path)


def save_comparison(
    original: np.ndarray,
    blurred: np.ndarray,
    restored: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Üç görüntüyü küçültmeden, her birini kendi deney çözünürlüğünde yan yana kaydeder."""
    height, width = original.shape[:2]
    header_height = 52
    canvas = Image.new("RGB", (width * 3, height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 5), title.replace("°", " deg"), fill="black")
    labels = ("Original", "Motion blurred", "Richardson-Lucy restored")
    for index, (array, label) in enumerate(zip((original, blurred, restored), labels)):
        draw.text((index * width + 10, 29), label, fill="black")
        image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8))
        canvas.paste(image, (index * width, header_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def get_input_images(use_demo: bool, max_side: int) -> list[tuple[str, np.ndarray]]:
    """Demo veya data/raw klasöründeki kendi net fotoğraflarını getirir."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if use_demo:
        demo = data.astronaut()
        return [("demo_astronaut", resize_rgb(demo, max_side))]
    paths = sorted(p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(
            f"{RAW_DIR} klasöründe görüntü bulunamadı. Önce buraya net JPG/PNG fotoğrafları ekle "
            "veya ilk deneme için 'python main.py --demo' komutunu çalıştır."
        )
    images: list[tuple[str, np.ndarray]] = []
    for path in paths:
        try:
            images.append((path.stem, load_rgb_image(path, max_side)))
        except Exception as exc:
            print(f"Atlandı: {path.name} ({exc})")
    if not images:
        raise RuntimeError("Klasördeki görüntüler okunamadı. JPG veya PNG ile dene.")
    return images


def run_experiment(
    use_demo: bool,
    blur_lengths: list[int],
    angles: list[float],
    iterations: int,
    max_side: int,
    method: str = "rl",
    tv_weight: float = 0.0012,
) -> None:
    """Bulanıklık üretir, Richardson-Lucy veya TV-RL ile geri kazanır, kalite ölçütlerini kaydeder."""
    if iterations < 1:
        raise ValueError("İterasyon sayısı en az 1 olmalı.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESTORED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    source_images = get_input_images(use_demo, max_side)
    rows: list[dict[str, float | int | str]] = []

    method_label = "TV-RL" if method == "tv-rl" else "Standart RL"
    print(f"Kaynak görüntü sayısı: {len(source_images)}")
    print(f"Görüntü uzun kenar sınırı: {max_side}px")
    print(
        f"Yöntem: {method_label}"
        + (f" (TV ağırlığı: {tv_weight})" if method == "tv-rl" else "")
    )
    print(f"Bulanıklık uzunlukları: {blur_lengths} | Açılar: {angles} | İterasyon: {iterations}")

    for image_index, (name, original) in enumerate(source_images, start=1):
        height, width = original.shape[:2]
        print(f"[{image_index}/{len(source_images)}] {name} ({width}x{height})")
        for length in blur_lengths:
            for angle in angles:
                psf = make_motion_psf(length, angle)
                blurred_padded, blurred_crop, pad = add_motion_blur_with_padding(original, psf, length)
                start = time.perf_counter()
                if method == "tv-rl":
                    restored_padded = restore_rgb_tv(blurred_padded, psf, iterations, tv_weight)
                else:
                    restored_padded = restore_rgb(blurred_padded, psf, iterations)
                processing_seconds = time.perf_counter() - start
                restored_crop = restored_padded[pad:pad + height, pad:pad + width]

                psnr_blurred = peak_signal_noise_ratio(original, blurred_crop, data_range=1.0)
                psnr_restored = peak_signal_noise_ratio(original, restored_crop, data_range=1.0)
                ssim_blurred = structural_similarity(original, blurred_crop, data_range=1.0, channel_axis=-1)
                ssim_restored = structural_similarity(original, restored_crop, data_range=1.0, channel_axis=-1)

                suffix = "_tv" if method == "tv-rl" else ""
                tag = f"{name}_len{length}_angle{int(angle)}_{width}x{height}_iter{iterations}{suffix}"
                comparison_path = OUTPUT_DIR / f"{tag}_comparison.png"
                restored_path = RESTORED_DIR / f"{tag}_restored.png"
                title = (
                    f"{name} | uzunluk={length}px | aci={angle} | iter={iterations} | "
                    + (f"TV-RL (lambda={tv_weight})" if method == "tv-rl" else "Standart RL")
                )
                save_comparison(
                    original,
                    blurred_crop,
                    restored_crop,
                    comparison_path,
                    title,
                )
                save_rgb(restored_crop, restored_path)
                rows.append({
                    "image": name,
                    "method": method,
                    "tv_weight": tv_weight if method == "tv-rl" else 0.0,
                    "width": width,
                    "height": height,
                    "blur_length_px": length,
                    "angle_degrees": angle,
                    "iterations": iterations,
                    "psnr_blurred_db": float(psnr_blurred),
                    "psnr_restored_db": float(psnr_restored),
                    "psnr_change_db": float(psnr_restored - psnr_blurred),
                    "ssim_blurred": float(ssim_blurred),
                    "ssim_restored": float(ssim_restored),
                    "ssim_change": float(ssim_restored - ssim_blurred),
                    "restoration_seconds": float(processing_seconds),
                    "comparison_image": str(comparison_path.relative_to(BASE_DIR)),
                    "restored_image": str(restored_path.relative_to(BASE_DIR)),
                })
                print(
                    f"  uzunluk={length:2d}, açı={angle:>5}° | "
                    f"PSNR: {psnr_blurred:.2f} -> {psnr_restored:.2f} dB | "
                    f"SSIM: {ssim_blurred:.4f} -> {ssim_restored:.4f} | "
                    f"İşlem: {processing_seconds:.2f} sn"
                )

    suffix = "_tv" if method == "tv-rl" else ""
    metrics_path = RESULTS_DIR / f"metrics_maxside{max_side}_iter{iterations}{suffix}.csv"
    if rows:
        with metrics_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (RESULTS_DIR / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print("\nTamamlandı.")
    print(f"Tam çözünürlüklü karşılaştırma görselleri: {OUTPUT_DIR}")
    print(f"Tek başına geri kazanılmış görüntüler: {RESTORED_DIR}")
    print(f"Metrik tablosu: {metrics_path}")
    print("Not: Sentetik bulanıklıkta PSF biliniyor; gerçek bulanık fotoğraflarda PSF ayrıca tahmin edilmelidir.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Richardson-Lucy hareket bulanıklığı giderme deneyi")
    parser.add_argument("--demo", action="store_true", help="Kod testi için yerleşik örnek görüntüyü kullan")
    parser.add_argument("--blur-lengths", nargs="+", type=int, default=[9, 17, 25],
                        help="Tek sayı biçiminde hareket bulanıklığı uzunlukları (varsayılan: 9 17 25)")
    parser.add_argument("--angles", nargs="+", type=float, default=[0, 45, 90],
                        help="Hareket yönleri derece cinsinden (varsayılan: 0 45 90)")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Richardson-Lucy iterasyon sayısı (varsayılan: 30)")
    parser.add_argument("--max-side", type=int, default=512,
                        help="En uzun görüntü kenarı sınırı (varsayılan: 512 px)")
    parser.add_argument("--method", choices=["rl", "tv-rl"], default="rl",
                        help="Kullanılacak algoritma: 'rl' (standart Richardson-Lucy) veya 'tv-rl' (Total Variation düzenlileştirmeli RL)")
    parser.add_argument("--tv-weight", type=float, default=0.0012,
                        help="TV-RL için Total Variation ağırlığı lambda (varsayılan: 0.0012)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        use_demo=args.demo,
        blur_lengths=args.blur_lengths,
        angles=args.angles,
        iterations=args.iterations,
        max_side=args.max_side,
        method=args.method,
        tv_weight=args.tv_weight,
    )


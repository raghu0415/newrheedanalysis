import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# --- CONFIGURATION ---
BASE_FOLDERS = {
    "mbe": "data/raw/mbe",
    "mbesubstrate": "data/raw/mbesubstrate",
    "test": "data/raw/test",
}
OUTPUT_DIR = "data/processed/physics_analysis"
DIAG_DIR = os.path.join(OUTPUT_DIR, "diagnostics")
TARGET_WIDTH = 1024
EPSILON = 1e-7  # Prevents division by zero errors


def quadratic_bg(x, a, b, c):
    return a * x**2 + b * x + c


def get_fwhm(y, peak_idx):
    """Calculates width at 50% height with a local window to avoid overlap."""
    half_max = y[peak_idx] / 2
    # Local window of 80 pixels around peak
    start, end = max(0, peak_idx - 40), min(len(y), peak_idx + 40)
    window = y[start:end]
    width_pts = np.where(window > half_max)[0]
    return float(len(width_pts)) if len(width_pts) > 0 else 0.0


def process_rheed(image_path, group, subgroup):
    # 1. Load and Standardize (B&W + 1024px)
    img_raw = cv2.imread(image_path, 0)
    if img_raw is None:
        return None

    img = cv2.resize(
        img_raw, (TARGET_WIDTH, TARGET_WIDTH), interpolation=cv2.INTER_AREA
    )
    h, w = img.shape
    mid_y = h // 2
    x = np.arange(w)

    # 2. Profile Extraction & Background Subtraction
    profile = np.mean(img[mid_y - 20 : mid_y + 20, :], axis=0)
    try:
        # Fit to 10% edges to model phosphor screen glow
        mask = (x < w * 0.1) | (x > w * 0.9)
        popt, _ = curve_fit(quadratic_bg, x[mask], profile[mask])
        bg = quadratic_bg(x, *popt)
        clean_y = np.maximum(profile - bg, 0)
    except:
        bg = np.zeros_like(x)
        clean_y = profile

    # 3. Peak Finding (Find 3 most central streaks)
    peaks, _ = find_peaks(clean_y, height=np.mean(clean_y) * 1.1, distance=w // 15)
    if len(peaks) < 1:
        return None

    # Sort by proximity to center, then sort left-to-right
    sorted_peaks = sorted(peaks, key=lambda p: abs(p - (w // 2)))[:3]
    sorted_peaks.sort()

    # Map FWHM to Left, Center, Right
    fwhms = {"L": None, "C": None, "R": None}
    if len(sorted_peaks) == 3:
        fwhms["L"], fwhms["C"], fwhms["R"] = [
            get_fwhm(clean_y, p) for p in sorted_peaks
        ]
    elif len(sorted_peaks) == 1:
        fwhms["C"] = get_fwhm(clean_y, sorted_peaks[0])
    else:  # 2 peaks found
        fwhms["L"] = get_fwhm(clean_y, sorted_peaks[0])
        fwhms["C"] = get_fwhm(clean_y, sorted_peaks[1])

    # 4. Metric Calculations
    # Sharpness = 100 / FWHM (Standardized to 1024px scale)
    main_fwhm = fwhms["C"] if fwhms["C"] else (fwhms["L"] if fwhms["L"] else EPSILON)
    sharpness = 100.0 / (main_fwhm + EPSILON)

    # Flatness (clamped 0-1) - Uses Coefficient of Variation
    peak_pos = sorted_peaks[1] if len(sorted_peaks) > 1 else sorted_peaks[0]
    v_prof = img[:, peak_pos].astype(float)
    cv = np.std(v_prof) / (np.mean(v_prof) + EPSILON)
    flatness = np.clip(1.0 - cv, 0.0, 1.0)

    # Symmetry Check (L vs R width difference)
    sym_error = abs(fwhms["L"] - fwhms["R"]) if (fwhms["L"] and fwhms["R"]) else 0.0

    # 5. Diagnostic Plotting (The Error Check)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.axhline(mid_y, color="red", alpha=0.3, ls="--")
    for p in sorted_peaks:
        plt.axvline(p, color="lime", alpha=0.6)
    plt.title(f"Image: {os.path.basename(image_path)}")

    plt.subplot(1, 2, 2)
    plt.plot(x, profile, color="gray", alpha=0.3, label="Raw")
    plt.plot(x, bg, "b--", label="Background")
    plt.plot(x, clean_y, "k", label="Cleaned")
    for p in sorted_peaks:
        plt.plot(p, clean_y[p], "rx")
    plt.title(f"Sharpness: {sharpness:.1f} | Flatness: {flatness:.2f}")
    plt.legend(prop={"size": 8})

    plt.savefig(os.path.join(DIAG_DIR, f"diag_{os.path.basename(image_path)}.jpg"))
    plt.close()

    return {
        "group": group,
        "subgroup": subgroup,
        "filename": os.path.basename(image_path),
        "fwhm_L": fwhms["L"],
        "fwhm_C": fwhms["C"],
        "fwhm_R": fwhms["R"],
        "sharpness_score": round(sharpness, 2),
        "flatness_score": round(flatness, 3),
        "symmetry_error": round(sym_error, 2),
    }


print(f"Current Working Directory: {os.getcwd()}")
print(f"Looking for MBE folder at: {os.path.abspath(BASE_FOLDERS['mbe'])}")

if __name__ == "__main__":
    for d in [OUTPUT_DIR, DIAG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    all_results = []
    for label, base_path in BASE_FOLDERS.items():
        if not os.path.exists(base_path):
            continue
        print(f"Analyzing folder: {label}")
        for root, _, files in os.walk(base_path):
            sub = os.path.relpath(root, base_path)
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    res = process_rheed(os.path.join(root, f), label, sub)
                    if res:
                        all_results.append(res)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUTPUT_DIR, "rheed_master_report.csv"), index=False)
    print(f"\nReport generated! Total images processed: {len(df)}")

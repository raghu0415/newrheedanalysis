import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
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
EPSILON = 1e-7


# --- MATHEMATICAL MODELS ---
def lorentzian(x, amp, ctr, hwhm):
    """Standard Lorentzian for single streaks."""
    return amp * hwhm**2 / ((x - ctr) ** 2 + hwhm**2)


def double_lorentzian(x, a1, c1, w1, a2, c2, w2):
    """Sum of two Lorentzians to handle saturated/split center peaks."""
    return lorentzian(x, a1, c1, w1) + lorentzian(x, a2, c2, w2)


def fit_zone(x_local, y_local, p_loc, is_center, is_substrate):
    """Fits the appropriate model to a local peak region."""
    try:
        if is_center and is_substrate:
            # Substrate Center: Doublet Fit
            # Seeds: [Amp1, Ctr1, Wid1, Amp2, Ctr2, Wid2]
            p0 = [np.max(y_local), p_loc - 15, 12, np.max(y_local), p_loc + 15, 12]
            popt, _ = curve_fit(double_lorentzian, x_local, y_local, p0=p0, maxfev=4000)
            # FWHM = Distance between centers + sum of HWHMs
            fwhm = abs(popt[1] - popt[4]) + (abs(popt[2]) + abs(popt[5]))
            return double_lorentzian(x_local, *popt), fwhm
        else:
            # Growth Streaks or Side Peaks: Single Fit
            p0 = [np.max(y_local), p_loc, 15]
            popt, _ = curve_fit(lorentzian, x_local, y_local, p0=p0, maxfev=3000)
            return lorentzian(x_local, *popt), abs(popt[2] * 2)
    except:
        # Fallback if fit fails to converge
        return np.zeros_like(x_local), 0.0


def process_rheed(image_path, group, subgroup):
    # 1. Image Prep
    img_raw = cv2.imread(image_path, 0)
    if img_raw is None:
        return None
    img = cv2.resize(img_raw, (TARGET_WIDTH, TARGET_WIDTH))
    filename = os.path.basename(image_path)
    is_substrate = ".3" in filename

    # 2. Advanced Background & Smoothing
    profile = np.mean(img[TARGET_WIDTH // 2 - 30 : TARGET_WIDTH // 2 + 30, :], axis=0)
    smoothed = savgol_filter(profile, 31, 3)
    # Pull the baseline to zero so Lorentzians can converge
    baseline = np.percentile(smoothed, 15 if is_substrate else 10)
    clean_y = np.maximum(smoothed - baseline, 0)

    # 3. Dynamic Peak Seeding
    # Look for 3 tallest peaks, min distance 100px apart
    peaks, props = find_peaks(clean_y, height=np.max(clean_y) * 0.15, distance=100)
    if len(peaks) < 1:
        return None
    best_peaks = sorted(peaks[np.argsort(props["peak_heights"])[-3:]])
    center_idx = np.argmin(np.abs(np.array(best_peaks) - 512))

    # 4. Fitting Logic
    fwhm_results = [0.0, 0.0, 0.0]
    visual_model = np.zeros(TARGET_WIDTH)

    for i, p_loc in enumerate(best_peaks):
        win = 85  # Window for local fit
        x_idx = np.arange(max(0, p_loc - win), min(TARGET_WIDTH, p_loc + win))
        curve, fwhm = fit_zone(
            x_idx, clean_y[x_idx], p_loc, (i == center_idx), is_substrate
        )
        visual_model[x_idx] += curve
        fwhm_results[i] = fwhm

    # 5. Flatness Calculation
    center_col = int(best_peaks[center_idx])
    v_prof = img[:, center_col].astype(float)
    flatness = np.clip(1.0 - (np.std(v_prof) / (np.mean(v_prof) + EPSILON)), 0, 1)

    # 6. Final Visualization (Verification)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    for p in best_peaks:
        plt.axvline(p, color="lime", ls="--", alpha=0.5)
    plt.title(f"{filename}")

    plt.subplot(1, 2, 2)
    plt.plot(clean_y, "k", alpha=0.3, label="Data")
    plt.plot(visual_model, "r", lw=2, label="Lorentz Model")
    plt.title(
        f"{'SUBSTRATE' if is_substrate else 'GROWTH'} | FWHM_C: {fwhm_results[1]:.1f}"
    )
    plt.legend(prop={"size": 8})

    plt.savefig(os.path.join(DIAG_DIR, f"diag_{filename}.jpg"))
    plt.close()

    return {
        "mode": "Substrate" if is_substrate else "Growth",
        "group": group,
        "filename": filename,
        "fwhm_L": round(fwhm_results[0], 2),
        "fwhm_C": round(fwhm_results[1], 2),
        "fwhm_R": round(fwhm_results[2], 2),
        "sharpness": round(100 / (fwhm_results[1] + EPSILON), 2),
        "flatness": round(flatness, 3),
    }


# --- BATCH EXECUTION ---
if __name__ == "__main__":
    for d in [OUTPUT_DIR, DIAG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    all_res = []
    for label, base_path in BASE_FOLDERS.items():
        if not os.path.exists(base_path):
            continue
        for root, _, files in os.walk(base_path):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    res = process_rheed(os.path.join(root, f), label, root)
                    if res:
                        all_res.append(res)

    df = pd.DataFrame(all_res)
    # Save separate reports for cleaner analysis
    df[df["mode"] == "Substrate"].to_csv(
        os.path.join(OUTPUT_DIR, "substrate_only.csv"), index=False
    )
    df[df["mode"] == "Growth"].to_csv(
        os.path.join(OUTPUT_DIR, "growth_only.csv"), index=False
    )
    print(
        "Done! Check 'substrate_only.csv' and 'growth_only.csv' in the processed folder."
    )

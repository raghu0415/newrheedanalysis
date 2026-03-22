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


# --- PHYSICS MATH MODELS ---
def lorentzian(x, amp, ctr, hwhm):
    return amp * hwhm**2 / ((x - ctr) ** 2 + hwhm**2)


def double_lorentzian(x, a1, c1, w1, a2, c2, w2):
    """Fits two peaks in the center to handle saturation/splitting."""
    return lorentzian(x, a1, c1, w1) + lorentzian(x, a2, c2, w2)


def process_rheed(image_path, group, subgroup):
    # 1. Load and Standardize
    img_raw = cv2.imread(image_path, 0)
    if img_raw is None:
        return None
    img = cv2.resize(
        img_raw, (TARGET_WIDTH, TARGET_WIDTH), interpolation=cv2.INTER_AREA
    )
    h, w = img.shape
    x = np.arange(w)

    # 2. Profile Extraction & Smoothing
    # We average the center band (40px) to reduce noise
    profile = np.mean(img[h // 2 - 20 : h // 2 + 20, :], axis=0)
    # Savgol filter helps locate real peaks in noisy MBE data (10.1)
    smoothed = savgol_filter(profile, 51, 3)
    clean_y = np.maximum(profile - np.percentile(profile, 15), 0)

    # 3. Dynamic Peak Seeding (Finding the 'Dominant 3')
    # Look for 3 tallest peaks with enough distance to be separate streaks
    peaks, props = find_peaks(smoothed, height=np.mean(smoothed) * 0.5, distance=120)
    if len(peaks) < 1:
        return None

    # Take the 3 tallest peaks and sort them Left-to-Right
    best_peaks = peaks[np.argsort(props["peak_heights"])[-3:]]
    best_peaks.sort()

    # Identify the Center peak (the one closest to the image middle 512)
    center_idx_in_list = np.argmin(np.abs(best_peaks - 512))

    # 4. Fitting Logic per Zone
    results = {"fwhm_L": 0, "fwhm_C": 0, "fwhm_R": 0}
    full_fit_visual = np.zeros_like(x)

    for i, p_loc in enumerate(best_peaks):
        # Window size for local fitting (120px wide)
        win = 60
        x_local = x[max(0, p_loc - win) : min(1024, p_loc + win)]
        y_local = clean_y[max(0, p_loc - win) : min(1024, p_loc + win)]

        try:
            if i == center_idx_in_list:
                # --- DOUBLE LORENTZIAN FOR CENTER (Handle Splitting) ---
                # p0: [Amp1, Ctr1, Wid1, Amp2, Ctr2, Wid2]
                p0 = [np.max(y_local), p_loc - 15, 10, np.max(y_local), p_loc + 15, 10]
                popt, _ = curve_fit(double_lorentzian, x_local, y_local, p0=p0)

                # Combined FWHM for a split peak is the span between the two peaks
                # plus the average HWHM of both.
                total_span = abs(popt[1] - popt[4]) + (abs(popt[2]) + abs(popt[5]))
                results["fwhm_C"] = total_span
                full_fit_visual[max(0, p_loc - win) : min(1024, p_loc + win)] += (
                    double_lorentzian(x_local, *popt)
                )
            else:
                # --- SINGLE LORENTZIAN FOR SIDES ---
                p0 = [np.max(y_local), p_loc, 15]
                popt, _ = curve_fit(lorentzian, x_local, y_local, p0=p0)
                fwhm = abs(popt[2] * 2)

                key = "fwhm_L" if i < center_idx_in_list else "fwhm_R"
                results[key] = fwhm
                full_fit_visual[max(0, p_loc - win) : min(1024, p_loc + win)] += (
                    lorentzian(x_local, *popt)
                )
        except Exception:
            continue

    # 5. Diagnostic Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray", aspect="auto")
    for p in best_peaks:
        plt.axvline(p, color="lime", ls="--", alpha=0.7)
    plt.title(f"Image: {os.path.basename(image_path)}")

    plt.subplot(1, 2, 2)
    plt.plot(x, clean_y, color="gray", alpha=0.4, label="Data")
    plt.plot(x, full_fit_visual, "r-", linewidth=2, label="Multi-Zone Fit")
    plt.title(
        f"C_FWHM: {results['fwhm_C']:.1f}px | Sharpness: {100 / (results['fwhm_C'] + EPSILON):.1f}"
    )
    plt.legend(prop={"size": 8})

    plt.savefig(os.path.join(DIAG_DIR, f"diag_{os.path.basename(image_path)}.jpg"))
    plt.close()

    # 6. Flatness Score (calculated on the center streak column)
    center_loc = int(best_peaks[center_idx_in_list])
    v_prof = img[:, center_loc].astype(float)
    flatness = np.clip(1.0 - (np.std(v_prof) / (np.mean(v_prof) + EPSILON)), 0, 1)

    return {
        "group": group,
        "subgroup": subgroup,
        "filename": os.path.basename(image_path),
        "fwhm_L": round(results["fwhm_L"], 2),
        "fwhm_C": round(results["fwhm_C"], 2),
        "fwhm_R": round(results["fwhm_R"], 2),
        "sharpness": round(100 / (results["fwhm_C"] + EPSILON), 2),
        "flatness": round(flatness, 3),
    }


if __name__ == "__main__":
    for d in [OUTPUT_DIR, DIAG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    final_data = []
    for label, base_path in BASE_FOLDERS.items():
        if not os.path.exists(base_path):
            continue
        print(f"Analyzing folder: {label}")
        for root, _, files in os.walk(base_path):
            sub_label = os.path.relpath(root, base_path)
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    res = process_rheed(os.path.join(root, f), label, sub_label)
                    if res:
                        final_data.append(res)

    df = pd.DataFrame(final_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "rheed_final_lorentz_report.csv"), index=False)
    print(f"\nReport created. Processed {len(df)} images.")

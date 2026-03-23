import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# --- PATHS ---
INPUT_DIR = "data/raw/mbesubstrate"
OUTPUT_DIR = "data/processed/physics_analysis/substrate"
DIAG_DIR = os.path.join(OUTPUT_DIR, "diagnostics")


def lorentzian(x, amp, ctr, hwhm):
    return amp * hwhm**2 / ((x - ctr) ** 2 + hwhm**2)


def double_lorentzian(x, a1, c1, w1, a2, c2, w2):
    return lorentzian(x, a1, c1, w1) + lorentzian(x, a2, c2, w2)


def process_substrate(path):
    img = cv2.imread(path, 0)
    if img is None:
        return None
    img = cv2.resize(img, (1024, 1024))

    # Aggressive background removal for high-contrast substrates
    profile = np.mean(img[480:540, :], axis=0)
    smoothed = savgol_filter(profile, 31, 3)
    clean_y = np.maximum(smoothed - np.percentile(smoothed, 20), 0)

    # Find widely spaced peaks (min distance 150px)
    peaks, props = find_peaks(clean_y, height=np.max(clean_y) * 0.15, distance=150)
    if len(peaks) < 1:
        return None
    best_peaks = sorted(peaks[np.argsort(props["peak_heights"])[-3:]])
    center_idx = np.argmin(np.abs(np.array(best_peaks) - 512))

    x = np.arange(1024)
    visual_fit = np.zeros(1024)
    res = {"fwhm_L": 0, "fwhm_C": 0, "fwhm_R": 0}

    for i, p_loc in enumerate(best_peaks):
        win = 100
        idx = np.arange(max(0, p_loc - win), min(1024, p_loc + win))
        x_loc, y_loc = x[idx], clean_y[idx]

        try:
            if i == center_idx:
                p0 = [np.max(y_loc), p_loc - 15, 10, np.max(y_loc), p_loc + 15, 10]
                popt, _ = curve_fit(double_lorentzian, x_loc, y_loc, p0=p0)
                visual_fit[idx] = double_lorentzian(x_loc, *popt)
                res["fwhm_C"] = abs(popt[1] - popt[4]) + (abs(popt[2]) + abs(popt[5]))
            else:
                p0 = [np.max(y_loc), p_loc, 15]
                popt, _ = curve_fit(lorentzian, x_loc, y_loc, p0=p0)
                visual_fit[idx] = lorentzian(x_loc, *popt)
                key = "fwhm_L" if i < center_idx else "fwhm_R"
                res[key] = abs(popt[2] * 2)
        except:
            continue

    plt.figure(figsize=(10, 4))
    plt.plot(clean_y, "k", alpha=0.3, label="Data")
    plt.plot(visual_fit, "r", lw=2, label="Doublet/Singlet Model")
    plt.title(f"Substrate Analysis: {os.path.basename(path)}")
    plt.savefig(os.path.join(DIAG_DIR, f"diag_{os.path.basename(path)}.jpg"))
    plt.close()

    return {"filename": os.path.basename(path), **res}


if __name__ == "__main__":
    os.makedirs(DIAG_DIR, exist_ok=True)
    results = [
        process_substrate(os.path.join(INPUT_DIR, f))
        for f in os.listdir(INPUT_DIR)
        if ".3" in f
    ]
    pd.DataFrame([r for r in results if r]).to_csv(
        os.path.join(OUTPUT_DIR, "substrate_results.csv"), index=False
    )
    print(f"Substrate logic complete. Images in {DIAG_DIR}")

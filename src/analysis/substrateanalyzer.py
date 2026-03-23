import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# --- USER CONFIGURATION ---
# List of image numbers (from XX.3.png) that require the Double-Lorentzian fit
DOUBLET_FILES = [11, 13, 14, 16, 17, 19, 20, 21, 25, 26, 28, 31]

INPUT_DIR = "data/raw/mbesubstrate"
OUTPUT_DIR = "data/processed/physics_analysis/substrate"
DIAG_DIR = os.path.join(OUTPUT_DIR, "diagnostics")


# --- MODELS WITH DYNAMIC BASELINE (y0) ---
def lorentzian(x, amp, ctr, hwhm, y0):
    """Lorentzian that sits on a variable baseline (y0)."""
    return (amp * hwhm**2 / ((x - ctr) ** 2 + hwhm**2)) + y0


def double_lorentzian(x, a1, c1, w1, a2, c2, w2, y0):
    """Double Lorentzian sharing a single background baseline."""
    return lorentzian(x, a1, c1, w1, 0) + lorentzian(x, a2, c2, w2, 0) + y0


def process_substrate(path):
    img_raw = cv2.imread(path, 0)
    if img_raw is None:
        return None
    img = cv2.resize(img_raw, (1024, 1024))
    filename = os.path.basename(path)

    try:
        file_num = int(filename.split(".")[0])
    except:
        file_num = -1

    # Profile extraction from center band
    profile = np.mean(img[480:540, :], axis=0)
    smoothed = savgol_filter(profile, 25, 3)

    # Identify the three main streaks
    peaks, _ = find_peaks(smoothed, prominence=8, distance=150)
    if len(peaks) < 3:
        peaks = np.argsort(smoothed)[-3:]

    best_peaks = sorted(peaks[np.argsort(smoothed[peaks])[-3:]])
    center_idx = np.argmin(np.abs(np.array(best_peaks) - 512))

    x = np.arange(1024)
    visual_fit = np.zeros(1024)
    # Added 'baseline' to the results dictionary to track it per image
    res = {
        "filename": filename,
        "fwhm_L": 0.0,
        "fwhm_C": 0.0,
        "fwhm_R": 0.0,
        "baseline": 0.0,
    }

    for i, p_loc in enumerate(best_peaks):
        win = 120
        idx = np.arange(max(0, p_loc - win), min(1024, p_loc + win))
        x_loc = x[idx]
        y_loc = smoothed[idx]

        # Identify the floor guess for this specific image
        floor_guess = np.min(y_loc)

        try:
            if i == center_idx and file_num in DOUBLET_FILES:
                # --- OFFSET DOUBLET FIT ---
                amp_seed = np.max(y_loc) - floor_guess
                p0 = [amp_seed, p_loc - 12, 15, amp_seed, p_loc + 12, 15, floor_guess]

                # Bounds allow the baseline to find its true level
                lower_b = [
                    amp_seed * 0.5,
                    p_loc - 40,
                    5,
                    amp_seed * 0.5,
                    p_loc + 2,
                    5,
                    0,
                ]
                upper_b = [
                    amp_seed * 3.0,
                    p_loc - 2,
                    65,
                    amp_seed * 2.5,
                    p_loc + 40,
                    65,
                    floor_guess + 50,
                ]

                popt, _ = curve_fit(
                    double_lorentzian,
                    x_loc,
                    y_loc,
                    p0=p0,
                    bounds=(lower_b, upper_b),
                    maxfev=5000,
                )
                visual_fit[idx] = double_lorentzian(x_loc, *popt)
                res["fwhm_C"] = round(
                    abs(popt[1] - popt[4]) + (abs(popt[2]) + abs(popt[5])), 2
                )
                res["baseline"] = round(popt[6], 2)  # Capture calculated baseline
            else:
                # --- OFFSET SINGLET FIT ---
                amp_seed = np.max(y_loc) - floor_guess
                p0 = [amp_seed, p_loc, 15, floor_guess]
                lower_b = [amp_seed * 0.5, p_loc - 35, 5, 0]
                upper_b = [amp_seed * 2.5, p_loc + 35, 65, floor_guess + 50]

                popt, _ = curve_fit(
                    lorentzian,
                    x_loc,
                    y_loc,
                    p0=p0,
                    bounds=(lower_b, upper_b),
                    maxfev=3000,
                )
                visual_fit[idx] = lorentzian(x_loc, *popt)
                key = ["fwhm_L", "fwhm_C", "fwhm_R"][i]
                res[key] = round(abs(popt[2] * 2), 2)
                if i == center_idx:
                    res["baseline"] = round(popt[3], 2)
        except:
            continue

    # Diagnostic plot showing the true baseline
    plt.figure(figsize=(10, 5))
    plt.plot(x, smoothed, color="gray", alpha=0.3, label="Raw Data")
    plt.plot(x, visual_fit, color="red", lw=2.5, label="Model Fit (with Baseline)")
    plt.title(f"{filename} | Calculated Baseline: {res['baseline']}")
    plt.legend()
    plt.savefig(os.path.join(DIAG_DIR, f"diag_{filename}.jpg"))
    plt.close()

    return res


if __name__ == "__main__":
    os.makedirs(DIAG_DIR, exist_ok=True)
    all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".png")])
    results = [process_substrate(os.path.join(INPUT_DIR, f)) for f in all_files]
    pd.DataFrame([r for r in results if r]).to_csv(
        os.path.join(OUTPUT_DIR, "final_substrate_results.csv"), index=False
    )
    print("Done. Check the CSV for the 'baseline' column to see the calculated values.")

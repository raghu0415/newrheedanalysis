import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# --- USER CONFIGURATION ---
# These are the specific file prefixes that require the Double-Lorentzian fit
DOUBLET_FILES = [11, 13, 14, 16, 17, 19, 20, 21, 25, 26, 28, 31]

INPUT_DIR = "data/raw/mbesubstrate"
OUTPUT_DIR = "data/processed/physics_analysis/substrate"
DIAG_DIR = os.path.join(OUTPUT_DIR, "diagnostics")


# --- MODELS ---
def lorentzian(x, amp, ctr, hwhm):
    return amp * hwhm**2 / ((x - ctr) ** 2 + hwhm**2)


def double_lorentzian(x, a1, c1, w1, a2, c2, w2):
    return lorentzian(x, a1, c1, w1) + lorentzian(x, a2, c2, w2)


def process_substrate(path):
    img = cv2.imread(path, 0)
    if img is None:
        return None
    img = cv2.resize(img, (1024, 1024))
    filename = os.path.basename(path)

    # Precise parsing for "11.3.png" -> 11
    try:
        file_num = int(filename.split(".")[0])
    except:
        file_num = -1

    profile = np.mean(img[480:540, :], axis=0)
    smoothed = savgol_filter(profile, 25, 3)

    # Catch peaks using prominence to handle background noise
    peaks, props = find_peaks(smoothed, prominence=8, distance=150)
    if len(peaks) < 3:
        peaks = np.argsort(smoothed)[-3:]

    best_peaks = sorted(peaks[np.argsort(smoothed[peaks])[-3:]])
    center_idx = np.argmin(np.abs(np.array(best_peaks) - 512))

    x = np.arange(1024)
    visual_fit = np.zeros(1024)
    res = {"filename": filename, "fwhm_L": 0.0, "fwhm_C": 0.0, "fwhm_R": 0.0}

    for i, p_loc in enumerate(best_peaks):
        win = 110
        idx = np.arange(max(0, p_loc - win), min(1024, p_loc + win))
        x_loc = x[idx]
        y_loc = smoothed[idx] - np.min(smoothed[idx])  # Local baseline correction

        try:
            # Force doublet logic for your specific list
            if i == center_idx and file_num in DOUBLET_FILES:
                p0 = [np.max(y_loc), p_loc - 12, 8, np.max(y_loc), p_loc + 12, 8]
                popt, _ = curve_fit(double_lorentzian, x_loc, y_loc, p0=p0, maxfev=5000)
                visual_fit[idx] = double_lorentzian(x_loc, *popt)
                res["fwhm_C"] = abs(popt[1] - popt[4]) + (abs(popt[2]) + abs(popt[5]))
            else:
                p0 = [np.max(y_loc), p_loc, 10]
                popt, _ = curve_fit(lorentzian, x_loc, y_loc, p0=p0, maxfev=3000)
                visual_fit[idx] = lorentzian(x_loc, *popt)
                key = ["fwhm_L", "fwhm_C", "fwhm_R"][i]
                res[key] = abs(popt[2] * 2)
        except:
            continue

    # Diagnostic output
    plt.figure(figsize=(10, 4))
    plt.plot(
        x, smoothed - np.percentile(smoothed, 10), color="gray", alpha=0.4, label="Data"
    )
    plt.plot(x, visual_fit, color="red", lw=2, label="Fit Model")
    mode_label = "DOUBLET" if file_num in DOUBLET_FILES else "STANDARD"
    plt.title(f"{filename} | Mode: {mode_label} | C_FWHM: {res['fwhm_C']:.1f}")
    plt.legend()
    plt.savefig(os.path.join(DIAG_DIR, f"diag_{filename}.jpg"))
    plt.close()

    return res


if __name__ == "__main__":
    os.makedirs(DIAG_DIR, exist_ok=True)
    all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".png")]
    results = [process_substrate(os.path.join(INPUT_DIR, f)) for f in all_files]

    pd.DataFrame([r for r in results if r]).to_csv(
        os.path.join(OUTPUT_DIR, "substrate_results_final.csv"), index=False
    )
    print(f"Finished. Diagnostics saved to {DIAG_DIR}")

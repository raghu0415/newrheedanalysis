import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# --- CONFIGURATION ---
DOUBLET_FILES = [11, 13, 14, 16, 17, 19, 20, 21, 25, 26, 28, 31]
INPUT_DIR = "data/raw/mbesubstrate"
OUTPUT_DIR = "data/processed/physics_analysis/substrate"
DIAG_DIR = os.path.join(OUTPUT_DIR, "diagnostics")


# --- MODELS ---
def lorentzian(x, amp, ctr, hwhm, y0):
    return (amp * hwhm**2 / ((x - ctr) ** 2 + hwhm**2)) + y0


def double_lorentzian(x, a1, c1, w1, a2, c2, w2, y0):
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

    profile = np.mean(img[480:540, :], axis=0)
    smoothed = savgol_filter(profile, 25, 3)

    # Hardened Peak Finding (Low prominence for clipped peaks)
    peaks, _ = find_peaks(smoothed, prominence=5, distance=120)
    if len(peaks) < 3:
        sorted_indices = np.argsort(smoothed)[::-1]
        fallback = []
        for idx in sorted_indices:
            if all(abs(idx - p) > 100 for p in fallback):
                fallback.append(idx)
            if len(fallback) == 3:
                break
        peaks = np.array(fallback)

    best_peaks = sorted(peaks[np.argsort(smoothed[peaks])[-3:]])
    center_idx = np.argmin(np.abs(np.array(best_peaks) - 512))
    peak_positions = []

    x = np.arange(1024)
    visual_fit = np.zeros(1024)

    # Initialize result container
    res = {
        "filename": filename,
        "fwhm_L": 0.0,
        "fwhm_C": 0.0,
        "fwhm_R": 0.0,
        "baseline": 0.0,
        "sharpness": 0.0,
        "symmetry_error": 0.0,
    }

    for i, p_loc in enumerate(best_peaks):
        win = 130
        idx = np.arange(max(0, p_loc - win), min(1024, p_loc + win))
        x_loc, y_loc = x[idx], smoothed[idx]
        floor_guess = np.min(y_loc)

        try:
            if i == center_idx and file_num in DOUBLET_FILES:
                # --- DOUBLET LOGIC ---
                amp_seed = np.max(y_loc) - floor_guess
                p0 = [amp_seed, p_loc - 12, 15, amp_seed, p_loc + 12, 15, floor_guess]
                popt, _ = curve_fit(double_lorentzian, x_loc, y_loc, p0=p0, maxfev=5000)
                visual_fit[idx] = double_lorentzian(x_loc, *popt)
                res["fwhm_C"] = round(
                    abs(popt[1] - popt[4]) + (abs(popt[2]) + abs(popt[5])), 2
                )
                res["baseline"] = round(popt[6], 2)
                res["sharpness"] = round(((popt[0] + popt[3]) / 2) / max(popt[6], 1), 2)
                peak_positions.append((popt[1] + popt[4]) / 2)
            else:
                # --- SINGLET LOGIC ---
                amp_seed = np.max(y_loc) - floor_guess
                p0 = [amp_seed, p_loc, 15, floor_guess]
                popt, _ = curve_fit(lorentzian, x_loc, y_loc, p0=p0, maxfev=3000)
                visual_fit[idx] = lorentzian(x_loc, *popt)
                key = ["fwhm_L", "fwhm_C", "fwhm_R"][i]
                res[key] = round(abs(popt[2] * 2), 2)
                peak_positions.append(popt[1])
                if i == center_idx:
                    res["baseline"] = round(popt[3], 2)
                    res["sharpness"] = round(popt[0] / max(popt[3], 1), 2)
        except:
            continue

    # Calculate Symmetry Error: |(Center-Left) - (Right-Center)|
    if len(peak_positions) == 3:
        dist_l = abs(peak_positions[1] - peak_positions[0])
        dist_r = abs(peak_positions[2] - peak_positions[1])
        res["symmetry_error"] = round(abs(dist_l - dist_r), 2)

    # Diagnostic Export
    plt.figure(figsize=(10, 5))
    plt.plot(x, smoothed, color="gray", alpha=0.3, label="Data")
    plt.plot(x, visual_fit, color="red", lw=2, label="Fit")
    plt.title(
        f"{filename} | Sharpness: {res['sharpness']} | SymmErr: {res['symmetry_error']}"
    )
    plt.savefig(os.path.join(DIAG_DIR, f"diag_{filename}.jpg"))
    plt.close()

    return res


if __name__ == "__main__":
    os.makedirs(DIAG_DIR, exist_ok=True)
    all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".png")])

    # Process and build list of dictionaries
    results_list = []
    for f in all_files:
        data = process_substrate(os.path.join(INPUT_DIR, f))
        if data:
            results_list.append(data)

    # Create the final CSV report
    final_df = pd.DataFrame(results_list)
    report_path = os.path.join(OUTPUT_DIR, "substrate_physics_report.csv")
    final_df.to_csv(report_path, index=False)

    print(f"Success! Final report generated at: {report_path}")

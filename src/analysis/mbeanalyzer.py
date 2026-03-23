import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# --- CONFIGURATION ---
INPUT_DIR = "data/raw/mbe"
OUTPUT_DIR = "data/processed/physics_analysis/growth"
DIAG_DIR = os.path.join(OUTPUT_DIR, "diagnostics")


def quadratic_bg(x, a, b, c):
    """Global background model for the entire RHEED profile."""
    return a * x**2 + b * x + c


def lorentzian(x, amp, ctr, hwhm):
    """Pure Lorentzian model (no baseline needed after subtraction)."""
    return amp * hwhm**2 / ((x - ctr) ** 2 + hwhm**2)


def process_growth_global_sub(path):
    img_raw = cv2.imread(path, 0)
    if img_raw is None:
        return None
    img = cv2.resize(img_raw, (1024, 1024))
    filename = os.path.basename(path)

    # Profile extraction: focus on the growth streak region
    profile = np.mean(img[500:520, :], axis=0)
    smoothed = savgol_filter(profile, 15, 3)
    x = np.arange(1024)

    # --- STEP 1: GLOBAL BACKGROUND ESTIMATION ---
    # Sampling the "valleys" (non-peak areas) to estimate diffuse scattering
    bg_indices = np.arange(0, 1024, 40)
    # Using a 10th percentile filter to find the noise floor effectively
    bg_points = [
        np.percentile(smoothed[max(0, i - 25) : min(1024, i + 25)], 10)
        for i in bg_indices
    ]

    try:
        popt_bg, _ = curve_fit(quadratic_bg, bg_indices, bg_points)
        full_bg = quadratic_bg(x, *popt_bg)
    except:
        # Fallback to a simple constant baseline if quadratic fails
        full_bg = np.full_like(x, np.percentile(smoothed, 10))

    # SUBTRACTION: This creates a 'flat' dataset for peak analysis
    flat_data = smoothed - full_bg
    flat_data[flat_data < 0] = 0  # No negative intensity allowed

    # --- STEP 2: ZONAL PEAK SEARCH ---
    # Define zones to ensure we get exactly -1, 0, and +1 orders
    zones = [(180, 420), (420, 600), (600, 820)]
    best_peaks = []
    for z_start, z_end in zones:
        zone_slice = flat_data[z_start:z_end]
        p, _ = find_peaks(zone_slice, prominence=2)
        if len(p) > 0:
            best_peaks.append(p[np.argmax(zone_slice[p])] + z_start)
        else:
            best_peaks.append(np.argmax(zone_slice) + z_start)

    # --- STEP 3: CLEAN LORENTZIAN FITTING ---
    visual_fit = np.zeros(1024)
    res = {
        "filename": filename,
        "fwhm_L": 0.0,
        "fwhm_C": 0.0,
        "fwhm_R": 0.0,
        "bg_curvature": 0.0,
    }

    for i, p_loc in enumerate(best_peaks):
        win = 85
        idx = np.arange(max(0, p_loc - win), min(1024, p_loc + win))
        x_loc, y_loc = x[idx], flat_data[idx]

        try:
            # Seed: [Amplitude, Center, HWHM]
            p0 = [np.max(y_loc), p_loc, 12]
            popt, _ = curve_fit(lorentzian, x_loc, y_loc, p0=p0)

            visual_fit[idx] = lorentzian(x_loc, *popt)
            key = ["fwhm_L", "fwhm_C", "fwhm_R"][i]
            res[key] = round(abs(popt[2] * 2), 2)
        except:
            continue

    # Store background curvature (a-parameter) for error checking
    res["bg_curvature"] = round(popt_bg[0], 6) if "popt_bg" in locals() else 0.0

    # --- STEP 4: TWO-STAGE DIAGNOSTIC PLOT ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Top Panel: Show original data and the background fit
    ax1.plot(x, smoothed, color="gray", alpha=0.5, label="Raw Intensity")
    ax1.plot(x, full_bg, color="red", ls="--", lw=2, label="Global Quadratic BG")
    ax1.set_title(f"BG Estimation: {filename}")
    ax1.legend()

    # Bottom Panel: Show 'Flat' data and Lorentzian fits
    ax2.plot(x, flat_data, color="black", alpha=0.3, label="BG-Subtracted Data")
    ax2.plot(x, visual_fit, color="lime", lw=2.5, label="Triple Lorentzian Fit")
    ax2.set_title(f"Clean Peak Fit (FWHM_C: {res['fwhm_C']})")
    ax2.set_ylim(0, np.max(flat_data) * 1.2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(DIAG_DIR, f"global_sub_{filename}.jpg"))
    plt.close()

    return res


if __name__ == "__main__":
    os.makedirs(DIAG_DIR, exist_ok=True)
    all_files = sorted(
        [
            f
            for f in os.listdir(INPUT_DIR)
            if f.lower().endswith(".png") and not f.endswith(".3.png")
        ]
    )

    results = [process_growth_global_sub(os.path.join(INPUT_DIR, f)) for f in all_files]
    pd.DataFrame([r for r in results if r]).to_csv(
        os.path.join(OUTPUT_DIR, "mbe_global_sub_report.csv"), index=False
    )
    print(
        f"Analysis complete. Check {DIAG_DIR} for the new two-stage diagnostic images."
    )

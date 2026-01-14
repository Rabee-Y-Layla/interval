"""
Laboratory Work #4
Interval Analysis with Jaccard Index
Complete version: computation + plots + printing + saving
"""

import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. DATA READING
# ==================================================

def read_bin_simple(filename, max_frames=50):
    voltages = []
    with open(filename, 'rb') as f:
        f.read(256)
        for _ in range(max_frames):
            f.read(16)
            raw = f.read(1024 * 8 * 2)
            if len(raw) < 1024 * 8 * 2:
                break
            data = np.frombuffer(raw, dtype=np.uint16)
            voltages.extend(data / 16384.0 - 0.5)
    return np.array(voltages)

# ==================================================
# 2. INTERVAL JACCARD INDEX
# ==================================================

def interval_jaccard(xl, xu, yl, yu):
    inter = np.maximum(np.minimum(xu, yu) - np.maximum(xl, yl), 0)
    union = np.maximum(xu, yu) - np.minimum(xl, yl)
    mask = union > 0
    j = np.zeros_like(union)
    j[mask] = inter[mask] / union[mask]
    return np.mean(j)

# ==================================================
# 3. INTERVAL STATISTICS
# ==================================================

def interval_mode(values, rad):
    hist, edges = np.histogram(values, bins=50)
    idx = np.argmax(hist)
    c = 0.5 * (edges[idx] + edges[idx + 1])
    return c - rad, c + rad

def kreinovich_median(lower, upper):
    return np.median(lower), np.median(upper)

def prolubnikov_median(centers, lower, upper):
    idx = np.argsort(centers)
    n = len(centers)
    if n % 2 == 1:
        i = idx[n // 2]
        return lower[i], upper[i]
    else:
        i1, i2 = idx[n // 2 - 1], idx[n // 2]
        return (lower[i1] + lower[i2]) / 2, (upper[i1] + upper[i2]) / 2

# ==================================================
# 4. MAIN ANALYSIS
# ==================================================

def analyze_complete(fileX, fileY):
    Xc = read_bin_simple(fileX)
    Yc = read_bin_simple(fileY)

    rad = 1 / 2**14
    Xl, Xu = Xc - rad, Xc + rad
    Yl, Yu = Yc - rad, Yc + rad

    # Interval estimates
    modeX = interval_mode(Xc, rad)
    modeY = interval_mode(Yc, rad)

    medKX = kreinovich_median(Xl, Xu)
    medKY = kreinovich_median(Yl, Yu)

    medPX = prolubnikov_median(Xc, Xl, Xu)
    medPY = prolubnikov_median(Yc, Yl, Yu)

    # Initial guesses
    a0 = np.mean(Yc) - np.mean(Xc)
    t0 = np.mean(Yc) / np.mean(Xc)

    a_vals = np.linspace(a0 - 0.02, a0 + 0.02, 200)
    t_vals = np.linspace(t0 - 0.05, t0 + 0.05, 200)

    # ------------------------------------------------
    # FUNCTIONALS
    # ------------------------------------------------

    F1a = [interval_jaccard(Xl + a, Xu + a, Yl, Yu) for a in a_vals]
    F2a = [interval_jaccard(modeX[0] + a, modeX[1] + a, modeY[0], modeY[1]) for a in a_vals]
    F3a = [interval_jaccard(medKX[0] + a, medKX[1] + a, medKY[0], medKY[1]) for a in a_vals]
    F4a = [interval_jaccard(medPX[0] + a, medPX[1] + a, medPY[0], medPY[1]) for a in a_vals]

    F1t = [interval_jaccard(np.minimum(Xl*t, Xu*t), np.maximum(Xl*t, Xu*t), Yl, Yu) for t in t_vals]
    F2t = [interval_jaccard(np.minimum(modeX[0]*t, modeX[1]*t), np.maximum(modeX[0]*t, modeX[1]*t), modeY[0], modeY[1]) for t in t_vals]
    F3t = [interval_jaccard(np.minimum(medKX[0]*t, medKX[1]*t), np.maximum(medKX[0]*t, medKX[1]*t), medKY[0], medKY[1]) for t in t_vals]
    F4t = [interval_jaccard(np.minimum(medPX[0]*t, medPX[1]*t), np.maximum(medPX[0]*t, medPX[1]*t), medPY[0], medPY[1]) for t in t_vals]

    functionals = {
        "F1_a": (a_vals, F1a),
        "F2_a": (a_vals, F2a),
        "F3_a": (a_vals, F3a),
        "F4_a": (a_vals, F4a),
        "F1_t": (t_vals, F1t),
        "F2_t": (t_vals, F2t),
        "F3_t": (t_vals, F3t),
        "F4_t": (t_vals, F4t),
    }

    # ------------------------------------------------
    # RESULTS: PRINT + SAVE + PLOTS
    # ------------------------------------------------

    results = {}

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    for name, (x, y) in functionals.items():
        idx = np.argmax(y)
        param_opt = x[idx]
        j_opt = y[idx]
        results[name] = (param_opt, j_opt)

        # Print
        print(f"{name:<6}: value = {param_opt:.6f}, Jaccard = {j_opt:.6f}")

        # Plot
        plt.figure()
        plt.plot(x, y, linewidth=2)
        plt.scatter(param_opt, j_opt, color='red', zorder=5)
        plt.xlabel("parameter")
        plt.ylabel("Jaccard index")
        plt.title(name)
        plt.grid(alpha=0.3)
        plt.savefig(f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Save results to file
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("Interval Analysis Results\n")
        f.write("="*50 + "\n")
        for k, v in results.items():
            f.write(f"{k}: value = {v[0]:.6f}, J = {v[1]:.6f}\n")

    print("\n✔ Results printed to console and saved to 'results.txt'")
    print("✔ All plots saved as PNG files\n")

# ==================================================
# 5. RUN
# ==================================================

if __name__ == "__main__":
    analyze_complete(
        "-0.205_lvl_side_a_fast_data.bin",
        "0.225_lvl_side_a_fast_data.bin"
    )
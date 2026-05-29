"""
=============================================================================
 MTH-203 Linear Algebra Project
 Topic  : Digital Image Compression using Singular Value Decomposition (SVD)
 Author : Abdul Rehman
 Date   : 2026-05-27
=============================================================================

Dependencies : numpy, matplotlib   (standard scientific Python stack)

Mathematical Background
-----------------------
Any real matrix A (m x n) can be factored as:

        A  =  U · Σ · Vᵀ

where:
  • U  (m x m)  – left-singular  vectors (orthonormal basis for column space)
  • Σ  (m x n)  – diagonal matrix of singular values σ₁ ≥ σ₂ ≥ … ≥ 0
  • Vᵀ (n x n)  – right-singular vectors (orthonormal basis for row space)

Rank-k Approximation:
  A_k = Σᵢ₌₁ᵏ  σᵢ · uᵢ · vᵢᵀ

  Instead of storing m×n numbers we only store:
      k·(m + 1 + n)  numbers  →  huge savings when k ≪ min(m,n)

=============================================================================
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")   # ensure Unicode symbols print on Windows
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all systems)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Load or generate the source image
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_PATH = "input_image.jpg"

def generate_synthetic_image_array(size: int = 400) -> np.ndarray:
    """
    Returns a (size × size) float64 NumPy array containing a synthetic
    grayscale geometric test image (values in [0, 255]).

    The image contains:
      • A soft radial gradient background
      • A filled circle
      • A filled rectangle
      • Diagonal stripes   (to exercise a richer singular-value spectrum)
      • Light Gaussian noise  (makes the SVD spectrum more realistic)
    """
    print(f"[INFO] Generating synthetic {size}×{size} grayscale image ...")

    rng    = np.random.default_rng(42)
    canvas = np.zeros((size, size), dtype=np.float64)

    y_idx, x_idx = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2

    # 1. Radial gradient background
    dist    = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    canvas += 180.0 * (1.0 - np.clip(dist / (size * 0.7), 0, 1))

    # 2. Filled circle
    canvas[dist < size * 0.25] = 220.0

    # 3. Filled rectangle
    r0, r1 = int(size * 0.55), int(size * 0.80)
    c0, c1 = int(size * 0.10), int(size * 0.45)
    canvas[r0:r1, c0:c1] = 60.0

    # 4. Diagonal stripes (adds higher-frequency content)
    canvas += 25.0 * (((x_idx + y_idx) % 20) / 20.0)

    # 5. Light Gaussian noise
    canvas += rng.normal(0, 5, canvas.shape)

    return np.clip(canvas, 0, 255)


# ── If a real image exists, try to load it; otherwise use synthetic data ─────
def load_image_as_matrix(path: str) -> np.ndarray:
    """
    Load any image file readable by matplotlib and convert to a 2-D
    float64 grayscale matrix with values in [0, 255].

    matplotlib.image.imread returns:
      • uint8  array (values 0–255)  for JPEG / BMP
      • float32 array (values 0–1)  for PNG
    We normalise to float64 [0–255] in both cases.
    """
    raw = mpimg.imread(path)          # (H, W) or (H, W, C)
    if raw.ndim == 3:
        # Convert RGB/RGBA → luminance via standard BT.601 weights
        weights = np.array([0.2989, 0.5870, 0.1140])
        gray = raw[..., :3] @ weights
    else:
        gray = raw.astype(np.float64)

    # Normalise to [0, 255] regardless of original dtype/scale
    if gray.max() <= 1.0:
        gray = gray * 255.0
    return gray.astype(np.float64)


if os.path.isfile(IMAGE_PATH):
    print(f"[INFO] Found '{IMAGE_PATH}' — loading it ...")
    try:
        A = load_image_as_matrix(IMAGE_PATH)
        print(f"[INFO] Loaded from file  →  shape = {A.shape}")
    except Exception as exc:
        print(f"[WARN] Could not load '{IMAGE_PATH}': {exc}")
        print("[INFO] Falling back to synthetic image.")
        A = generate_synthetic_image_array(400)
else:
    print(f"[INFO] '{IMAGE_PATH}' not found — using synthetic image.")
    A = generate_synthetic_image_array(400)

    # Optionally persist as PNG (no Pillow needed — matplotlib can write PNG)
    png_path = "input_image.png"
    mpimg.imsave(png_path, A, cmap="gray", vmin=0, vmax=255)
    print(f"[INFO] Synthetic image also saved as '{png_path}'")

m, n = A.shape
print(f"[INFO] Image matrix A  →  shape = {m} × {n},  dtype = {A.dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Compute the full SVD:  A = U · S · Vt
# ─────────────────────────────────────────────────────────────────────────────

print("\n[INFO] Computing full SVD of A ...")
U, S, Vt = np.linalg.svd(A, full_matrices=True)
# U  : (m × m)  orthogonal — left  singular vectors
# S  : (r,)     singular values σ₁ ≥ σ₂ ≥ … ≥ 0,  r = min(m,n)
# Vt : (n × n)  orthogonal — right singular vectors (already transposed)

r = min(m, n)
print(f"[INFO] SVD complete  →  U{U.shape}, S({S.shape[0]},), Vt{Vt.shape}")
print(f"[INFO] σ₁ = {S[0]:.2f}   |   σ_min = {S[-1]:.4f}")
print(f"[INFO] Energy in top-50 components: "
      f"{100 * np.sum(S[:50]**2) / np.sum(S**2):.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def rank_k_approximation(U, S, Vt, k):
    """
    Efficient rank-k reconstruction:
        A_k = U[:, :k] · diag(S[:k]) · Vt[:k, :]

    This is mathematically identical to the outer-product sum:
        Σᵢ₌₁ᵏ  σᵢ · uᵢ · vᵢᵀ
    but implemented as two matrix multiplications for speed.
    """
    k = min(k, len(S))
    return (U[:, :k] * S[:k]) @ Vt[:k, :]    # broadcasting avoids diag()


def storage_savings(m, n, k):
    """
    Compare storage cost of rank-k SVD vs. full pixel matrix.

    Full matrix  : m × n
    Rank-k store : k(m + 1 + n)
    """
    original   = m * n
    compressed = k * (m + 1 + n)
    pct        = max(0.0, (1.0 - compressed / original) * 100.0)
    return pct, original, compressed


def frobenius_error(A, A_k):
    """
    ‖A − A_k‖_F  =  sqrt( Σᵢⱼ (aᵢⱼ − a_k_ᵢⱼ)² )

    By Eckart–Young this is the minimum possible Frobenius error
    for *any* rank-k matrix approximation.
    """
    return float(np.linalg.norm(A - A_k, ord="fro"))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Run compression for three rank values
# ─────────────────────────────────────────────────────────────────────────────

K_VALUES   = [5, 30, 100]
compressed = {}   # k  →  uint8 pixel matrix

print("\n" + "=" * 68)
print(f"{'k':>6}  {'‖A−Aₖ‖_F':>18}  {'Savings':>10}  {'Elements (orig → comp)'}")
print("=" * 68)

for k in K_VALUES:
    A_k   = rank_k_approximation(U, S, Vt, k)
    error = frobenius_error(A, A_k)
    pct, orig_el, comp_el = storage_savings(m, n, k)
    compressed[k] = np.clip(A_k, 0, 255).astype(np.uint8)

    print(f"  k={k:<4d}  ‖error‖_F = {error:>12.4f}  "
          f"savings = {pct:>7.2f}%   "
          f"({orig_el:,} → {comp_el:,})")

print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Singular-value energy curve (used in annotations)
# ─────────────────────────────────────────────────────────────────────────────

cumulative_energy = np.cumsum(S**2) / np.sum(S**2) * 100


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – 2×2 comparison grid  →  compression_results.png
# ─────────────────────────────────────────────────────────────────────────────

CMAP        = "gray"
OUTPUT_FILE = "compression_results.png"

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    "Digital Image Compression via SVD\n"
    "MTH-203 Linear Algebra Project",
    fontsize=15, fontweight="bold", y=1.01
)

# ── Original ──────────────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.imshow(A, cmap=CMAP, vmin=0, vmax=255)
ax.set_title(
    f"Original Image\n(Full rank  |  {m}×{n} = {m*n:,} values)",
    fontsize=11, fontweight="bold"
)
ax.axis("off")

# ── Compressed versions ───────────────────────────────────────────────────────
positions = [(0, 1), (1, 0), (1, 1)]

for (row, col), k in zip(positions, K_VALUES):
    ax    = axes[row][col]
    error = frobenius_error(A, compressed[k].astype(np.float64))
    pct, _, _ = storage_savings(m, n, k)
    energy_k  = cumulative_energy[k - 1]

    ax.imshow(compressed[k], cmap=CMAP, vmin=0, vmax=255)
    ax.set_title(
        f"Rank-{k} Approximation\n"
        f"‖Error‖_F = {error:.1f}   |   Savings = {pct:.1f}%\n"
        f"Energy captured = {energy_k:.2f}%",
        fontsize=10
    )
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
print(f"\n[INFO] Comparison plot saved → '{OUTPUT_FILE}'")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – Singular-value spectrum plot  →  sv_spectrum.png
# ─────────────────────────────────────────────────────────────────────────────

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle("Singular Value Analysis", fontsize=13, fontweight="bold")

n_show = min(200, r)

# Log-scale magnitude
ax1.semilogy(S[:n_show], color="steelblue", linewidth=1.5)
ax1.set_xlabel("Index  i")
ax1.set_ylabel("σᵢ  (log scale)")
ax1.set_title(f"Singular Value Magnitude (first {n_show})")
ax1.grid(True, alpha=0.3)
for k in K_VALUES:
    ax1.axvline(x=k, linestyle="--", alpha=0.7, label=f"k={k}")
ax1.legend(fontsize=9)

# Cumulative energy
ax2.plot(cumulative_energy[:n_show], color="darkorange", linewidth=1.5)
ax2.set_xlabel("Number of singular values  k")
ax2.set_ylabel("Cumulative energy  (%)")
ax2.set_title("Cumulative Energy Captured")
ax2.grid(True, alpha=0.3)
for k in K_VALUES:
    if k <= n_show:
        ax2.axvline(x=k, linestyle="--", alpha=0.7)
        ax2.axhline(y=cumulative_energy[k - 1], linestyle=":", alpha=0.5)
ax2.set_ylim(0, 101)

plt.tight_layout()
sv_output = "sv_spectrum.png"
plt.savefig(sv_output, dpi=150, bbox_inches="tight")
print(f"[INFO] Spectrum plot saved    → '{sv_output}'")
plt.close()

print("\n✅  All done — check 'compression_results.png' and 'sv_spectrum.png'")

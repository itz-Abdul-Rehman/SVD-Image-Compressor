# Digital Image Compression via Singular Value Decomposition

**Course:** MTH-203 — Linear Algebra  
**Date:** 2026-05-27  
**Author:** *(Your Name)*

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)  
2. [Mathematical Modeling](#2-mathematical-modeling)  
3. [Algorithm & Implementation](#3-algorithm--implementation)  
4. [Results & Interpretation](#4-results--interpretation)  
5. [Conclusion](#5-conclusion)  
6. [References](#6-references)  

---

## 1. Problem Statement

### Why Does Image Compression Matter?

In the modern digital world, images are everywhere — social media, medical imaging, satellite data, video streaming. A single uncompressed 4K photo can be 25–50 MB. Now multiply that by billions of uploads per day. The problems this creates are real:

| Challenge | Real-World Impact |
|-----------|------------------|
| **Bandwidth** | Slow page loads, buffering video, expensive mobile data |
| **Storage** | Cloud storage costs, limited device capacity |
| **Transmission** | MRI scans sent between hospitals, satellite imagery downloads |
| **Archiving** | Museums digitizing thousands of artworks |

The goal of image compression is to represent an image using **fewer numbers** while keeping it **visually indistinguishable** (or acceptably similar) to the original.

### How Is an Image a Matrix?

A grayscale digital image is, at its core, a **rectangular grid of pixel intensity values**.  
- Each pixel holds a single number in the range **0–255** (0 = black, 255 = white).  
- A 400×400 grayscale image is simply a **400×400 matrix A** containing 160,000 numbers.  
- A color (RGB) image has three such matrices — one per channel.

This mathematical representation means we can apply powerful tools from **Linear Algebra** — specifically **Singular Value Decomposition** — to compress the data.

---

## 2. Mathematical Modeling

### 2.1 The SVD Formula

For any real matrix **A** of size *m × n*, the Singular Value Decomposition states that:

$$\boxed{A = U \cdot \Sigma \cdot V^T}$$

This is not an approximation — it is an *exact* factorization that always exists.

### 2.2 What Does Each Component Mean?

#### U — Left Singular Vectors (Vertical / Column Features)
- **Shape:** *m × m*
- **Orthogonal matrix:** its columns are mutually perpendicular unit vectors.
- **Meaning:** Each column **uᵢ** is a "vertical building block" of the image — it captures patterns along the **rows** (think vertical structure, gradients top-to-bottom).
- **Property:** U^T · U = I (the identity matrix)

#### Σ (Sigma) — The Diagonal of Singular Values (The Importance Weights)
- **Shape:** *m × n* (but only the diagonal entries matter)
- **Entries:** σ₁ ≥ σ₂ ≥ σ₃ ≥ … ≥ 0  (always non-negative, always sorted descending)
- **Meaning:** Each σᵢ is a **weight** telling us how important the corresponding pair of vectors (uᵢ, vᵢ) is in reconstructing A. A large σ₁ means that direction carries a lot of the image's "energy." A tiny σ₅₀₀ means that direction contributes almost nothing.
- **Key insight:** The first few singular values typically account for 90–99% of the total energy of the image.

#### Vᵀ — Right Singular Vectors (Horizontal / Row Features)
- **Shape:** *n × n*
- **Orthogonal matrix:** V^T · V = I
- **Meaning:** Each row **vᵢᵀ** is a "horizontal building block" — it captures patterns along the **columns** (horizontal structure, left-to-right gradients).

### 2.3 The Rank-k Approximation — How Compression Works

The full SVD can be written as a sum of **rank-1 matrices**:

$$A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^T + \cdots + \sigma_r \mathbf{u}_r \mathbf{v}_r^T$$

where *r = min(m, n)*.

The **rank-k approximation** simply **truncates this sum** after k terms:

$$\boxed{A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T}$$

By the **Eckart–Young Theorem**, this is the *best possible* rank-k approximation of A in terms of Frobenius norm — no other rank-k matrix is closer to A.

### 2.4 Storage Savings — The Mathematics

| What we store | Number of values |
|---------------|-----------------|
| **Full matrix A** | m × n |
| **Rank-k SVD** | k·m (U columns) + k (singular values) + k·n (Vᵀ rows) = **k(m + 1 + n)** |

**Compression ratio:**

$$\text{Storage Savings} = \left(1 - \frac{k(m+1+n)}{m \cdot n}\right) \times 100\%$$

**Example** — 400×400 image, k = 30:  
- Original: 160,000 values  
- Compressed: 30 × (400 + 1 + 400) = 24,030 values  
- **Savings ≈ 85%** 🎉

### 2.5 Measuring Data Loss — The Frobenius Norm

To quantify how much information we lost, we compute the **Frobenius norm of the error matrix**:

$$\|A - A_k\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} (a_{ij} - a_{k,ij})^2}$$

This is the square root of the sum of squared pixel differences — essentially the "total pixel error" across the entire image. A smaller Frobenius error means a more faithful reconstruction.

A beautiful result from SVD theory gives us the error analytically:

$$\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}$$

So the error is **entirely determined by the discarded singular values**.

---

## 3. Algorithm & Implementation

```
Input: Grayscale image → 2D NumPy matrix A (m × n)

Step 1: U, S, Vt = numpy.linalg.svd(A, full_matrices=True)
          → U  : (m×m)  left singular vectors
          → S  : (r,)   singular values, r = min(m,n)
          → Vt : (n×n)  right singular vectors (transposed)

Step 2: For each k in [5, 30, 100]:
    A_k  = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]  (rank-k reconstruction)
    error = ‖A − A_k‖_F                          (Frobenius norm)
    savings = (1 − k(m+1+n)/(m·n)) × 100%       (storage metric)

Step 3: Display and save comparison plot (compression_results.png)
```

**Libraries used:**
- `numpy` — matrix operations, SVD
- `Pillow (PIL)` — image loading / saving
- `matplotlib` — visualization

---

## 4. Results & Interpretation

> 📌 **Run `python image_compressor.py` and fill in the exact values below from the terminal output.**

### 4.1 Quantitative Results Table

| Rank k | Frobenius Error ‖A−Aₖ‖_F | Storage Savings (%) | Energy Captured (%) |
|--------|--------------------------|---------------------|---------------------|
| k = 5  | *(insert from script)*   | *(insert)*          | *(insert)*          |
| k = 30 | *(insert from script)*   | *(insert)*          | *(insert)*          |
| k = 100| *(insert from script)*   | *(insert)*          | *(insert)*          |

### 4.2 Visual Interpretation

Insert the saved image `compression_results.png` here:

![Compression Results](compression_results.png)

**What to observe:**
- **k = 5:** Very blurry — only the coarsest structure (overall brightness, dominant gradients) is visible. Frobenius error is very high, but storage savings are dramatic.
- **k = 30:** Noticeably cleaner. Major shapes and edges are recognizable. A good balance of quality vs. size.
- **k = 100:** Nearly indistinguishable from the original to the human eye. Error is very low. Still achieves significant storage savings.

### 4.3 Singular Value Spectrum

Insert `sv_spectrum.png` here:

![Singular Value Spectrum](sv_spectrum.png)

The rapid decay of singular values confirms that most of the image's information is concentrated in a **small number of dominant components** — which is exactly why SVD-based compression works so effectively on natural images.

---

## 5. Conclusion

This project demonstrated that **Singular Value Decomposition is a powerful and mathematically rigorous tool for image compression**:

- SVD decomposes any matrix into orthogonal building blocks ranked by importance.
- Truncating to rank-k gives the *optimal* low-rank approximation (Eckart–Young Theorem).
- Even with k = 30 (out of 400 possible components), the image remains visually clear while using ~85% less storage.
- The Frobenius norm provides a precise, objective measure of data loss — essential for engineering trade-off decisions.

The same mathematics underlies recommender systems (Netflix SVD), Principal Component Analysis (PCA), natural language processing (Latent Semantic Analysis), and noise reduction in signal processing.

---

## 6. References

1. Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.  
2. Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.  
3. Eckart, C., & Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika, 1*(3), 211–218.  
4. NumPy documentation — `numpy.linalg.svd`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html

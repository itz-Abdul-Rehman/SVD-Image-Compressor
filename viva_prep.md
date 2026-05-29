# 🎓 Viva Cheat Sheet — SVD Image Compression (MTH-203)

> **How to use this:** Read each question out loud, cover the answer, and try to say it yourself.  
> These are the 5 *hardest* questions your professor is likely to ask. Nail these and you're golden.

---

## Q1 — "What exactly IS the Singular Value Decomposition? Why does it always exist?"

**Answer:**

SVD says that *any* real matrix A — no matter the shape — can be written as:

> **A = U · Σ · Vᵀ**

where U and V are orthogonal matrices (their columns are perpendicular unit vectors), and Σ is diagonal with non-negative entries called **singular values**, sorted largest to smallest.

It always exists because the matrix **AᵀA** is symmetric and positive semi-definite — so by the Spectral Theorem it always has real, non-negative eigenvalues. The singular values σᵢ are simply the square roots of those eigenvalues. Since every real symmetric matrix has an orthogonal eigen-decomposition, SVD always exists for any real matrix. Full stop.

---

## Q2 — "Why is the rank-k approximation the *best* possible? Could you find a better one?"

**Answer:**

No — you cannot do better. This is guaranteed by the **Eckart–Young Theorem (1936)**:

> "Among all matrices of rank ≤ k, the truncated SVD gives the one closest to A in Frobenius norm (and also in spectral norm)."

**Intuition:** The singular values σ₁ ≥ σ₂ ≥ … are ordered by importance — σ₁ captures the most "energy" of the matrix, σ₂ the next most, and so on. When we keep only the top k, we are keeping the k most important directions and discarding the rest. Any other rank-k matrix will miss something more important.

**Proof sketch:** The error of the rank-k truncation is ‖A − Aₖ‖_F = √(σ²ₖ₊₁ + … + σ²ᵣ). To reduce this further you would need to remove a singular value *smaller* than σₖ — but you've already discarded the smallest ones. There's nothing left to cut without doing worse.

---

## Q3 — "What does it mean for U and V to be orthogonal, and why does that matter for this application?"

**Answer:**

A matrix is **orthogonal** if its columns are:
1. **Unit vectors** (each has length 1)
2. **Mutually perpendicular** (any two columns have dot product 0)

This is written as **UᵀU = UUᵀ = I**.

**Why it matters here:**

- **No distortion:** Orthogonal matrices preserve lengths and angles — they are pure rotations (or reflections). This means U and V don't stretch or shrink the image data; only Σ does the scaling.
- **Numerically stable:** Orthogonal matrices have condition number = 1. Computations with them don't amplify floating-point errors.
- **Perfect inversion for free:** U⁻¹ = Uᵀ. This means reconstruction is cheap — no matrix inversion needed.
- **Independent components:** Because the columns of U are orthogonal, each "image building block" uᵢ contributes independently. We can drop the small ones without affecting the large ones at all.

---

## Q4 — "What is the 'rank' of a matrix, and what does it tell us about an image?"

**Answer:**

The **rank** of matrix A is the number of **linearly independent rows (= columns)**.  
Equivalently, it equals the number of **non-zero singular values**.

In image terms:
- A **high-rank** image has complex, detailed structure — lots of independent pixel patterns (e.g., a photo of a city with fine textures).
- A **low-rank** image has repetitive or smooth structure (e.g., a solid grey rectangle is rank-1; a horizontal gradient is rank-1).

**Why this matters for compression:**

Natural images are not literally low-rank, but their singular values decay rapidly — the first few σᵢ are huge and the rest are tiny. This means A is "approximately low-rank," so a small k can capture most of the visual content. A rank-k approximation works better (with less error) on images whose singular values drop off quickly.

**One-liner for the professor:** "Rank measures how many truly independent pieces of information the matrix contains. SVD finds and ranks those pieces by importance."

---

## Q5 — "You mentioned a 'basis change' — can you explain what SVD has to do with change of basis?"

**Answer:**

Absolutely. SVD is, at its heart, a **change of basis theorem**.

The standard way we represent an image matrix A is in the **standard pixel basis** — each entry aᵢⱼ is just "the pixel at row i, column j." This basis is convenient but carries no notion of importance.

SVD finds a **new, better basis** for the same data:

1. **V columns** (rows of Vᵀ) form a new orthonormal basis for the **input space** (ℝⁿ, the columns of A).
2. **U columns** form a new orthonormal basis for the **output space** (ℝᵐ, the rows of A).
3. In this new basis, A becomes the diagonal matrix Σ — the simplest possible form.

**The compression insight:**
In the standard pixel basis, *every* pixel is equally important. In the SVD basis, importance is explicit — σ₁ is the most important direction, σ₂ the next, etc. So we can confidently **zero out the unimportant directions** (small σᵢ) while keeping the important ones. We couldn't do this in the pixel basis because no single pixel is obviously "unimportant."

**Analogy:** It's like rotating a messy cloud of data points until the longest axis of the cloud lines up with the x-axis, the second-longest with the y-axis, etc. Once aligned, it's obvious which directions matter — and SVD is the rotation that does this for matrices.

---

## ⚡ Lightning Round — Quick-Fire Definitions

| Term | One-sentence answer |
|------|---------------------|
| **Frobenius Norm** | Square root of the sum of all squared matrix entries — total "magnitude" of a matrix |
| **Singular Values** | Non-negative square roots of the eigenvalues of AᵀA, measuring each component's importance |
| **Low-Rank Approximation** | Approximating a matrix using fewer independent components to save storage |
| **Orthonormal Basis** | A set of mutually perpendicular unit vectors that span the space |
| **Eckart–Young Theorem** | Truncated SVD is the best low-rank approximation in both Frobenius and spectral norms |
| **Energy (in SVD)** | Proportion of Σσᵢ² explained by the top-k singular values — measures reconstruction quality |

---

## 🔑 Three Sentences to Memorize

1. *"SVD decomposes any matrix into a rotation, a scaling, and another rotation — U rotates, Σ scales, Vᵀ rotates."*
2. *"By Eckart–Young, keeping the top-k singular values gives the optimal rank-k approximation — we are discarding the least important structural directions."*
3. *"The Frobenius error of a rank-k approximation equals the square root of the sum of the squares of all discarded singular values — so we can predict the error analytically before even running the reconstruction."*

---

*Good luck in your viva! 🚀*

"""
MTH-203 Linear Algebra Project
SVD Image Compressor — Dashboard UI
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SVD Image Compressor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Background ── */
[data-testid="stAppViewContainer"] {
    background-color: #f0efeb;
}
[data-testid="stHeader"] {
    background-color: #f0efeb;
}
section[data-testid="stSidebar"] {
    background-color: #f0efeb;
}

/* ── Hide Streamlit branding and toolbar ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Card style ── */
.card {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border: 1px solid #e8e8e4;
}
.card-sm {
    background: #ffffff;
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid #e8e8e4;
    height: 100%;
}

/* ── Metric card ── */
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888880;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #1a1a18;
    line-height: 1.1;
}
.metric-unit {
    font-size: 14px;
    font-weight: 400;
    color: #888880;
    margin-left: 2px;
}
.metric-sub {
    font-size: 12px;
    color: #aaa;
    margin-top: 6px;
}

/* ── Section title ── */
.section-title {
    font-size: 15px;
    font-weight: 700;
    color: #1a1a18;
    margin-bottom: 14px;
}

/* ── Page title ── */
.page-title {
    font-size: 22px;
    font-weight: 700;
    color: #1a1a18;
    text-align: center;
    padding: 18px 0 4px 0;
    letter-spacing: -0.01em;
}
.page-sub {
    font-size: 13px;
    color: #888880;
    text-align: center;
    margin-bottom: 24px;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background-color: #1a9e8f !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 0 !important;
    width: 100% !important;
    transition: background 0.2s;
}
.stButton > button[kind="primary"]:hover {
    background-color: #178070 !important;
}

/* ── Secondary button ── */
.stButton > button[kind="secondary"] {
    background-color: #ffffff !important;
    color: #1a1a18 !important;
    border: 1.5px solid #d0d0cc !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background-color: #1a9e8f !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
    font-size: 13px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #178070 !important;
}

/* ── Sliders — track color ── */
[data-testid="stSlider"] > div > div > div > div {
    background-color: #1a9e8f !important;
}
/* Slider labels (k1 — Aggressive, etc.) */
[data-testid="stSlider"] label,
[data-testid="stSlider"] label p {
    color: #1a1a18 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
/* Hide the white tick-bar min/max boxes */
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] {
    display: none !important;
}
/* Current value label on slider */
[data-testid="stSlider"] [data-testid="stTickBar"] {
    display: none !important;
}
div[data-baseweb="slider"] [data-testid="stThumbValue"],
div[data-baseweb="slider"] span {
    color: #1a1a18 !important;
    background: transparent !important;
}

/* ── File uploader dropzone ── */
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed #ccc !important;
    border-radius: 10px !important;
    background: #fafaf8 !important;
}
/* Upload button — keep visible on hover */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploaderDropzone"] button:focus {
    background-color: #1a1a18 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    opacity: 1 !important;
    visibility: visible !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #333330 !important;
}

/* ── Stats row ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #f0efeb;
    font-size: 13px;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: #555550; }
.stat-val { font-weight: 700; color: #1a1a18; }
.stat-savings { color: #1a9e8f; font-weight: 700; }

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #e8e8e4;
    margin: 16px 0;
}
</style>
""", unsafe_allow_html=True)


# ── SVD Core Functions ────────────────────────────────────────────────────────

@st.cache_data
def load_and_svd(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    A   = np.array(img, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    return A, U, S, Vt

def load_as_grayscale(file_obj) -> np.ndarray:
    img = Image.open(file_obj).convert("L")
    return np.array(img, dtype=np.float64)

def rank_k_approx(U, S, Vt, k):
    k = min(k, len(S))
    return (U[:, :k] * S[:k]) @ Vt[:k, :]

def storage_savings(m, n, k):
    orig = m * n
    comp = k * (m + 1 + n)
    pct  = max(0.0, (1.0 - comp / orig) * 100.0)
    return pct, orig, comp

def frobenius_error(A, A_k):
    return float(np.linalg.norm(A - A_k, ord="fro"))

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">SVD Image Compressor</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Compress images using Singular Value Decomposition — MTH-203 Linear Algebra</div>', unsafe_allow_html=True)

# ── Layout: Left panel | Right panel ─────────────────────────────────────────
left, right = st.columns([1, 2.4], gap="large")

with left:
    # Upload card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Select an image from your computer",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Settings card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Rank-k Settings</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#888880;margin-bottom:14px">Lower k = more compressed. Higher k = better quality.</div>', unsafe_allow_html=True)

    k1 = st.slider("k1  — Aggressive", 1, 50, 5)
    k2 = st.slider("k2  — Balanced",   1, 150, 30)
    k3 = st.slider("k3  — Fine",       1, 300, 100)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    if uploaded is None:
        st.markdown("""
        <div class="card" style="min-height:300px;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:10px;">
            <div style="font-size:15px;font-weight:600;color:#1a1a18">No results yet</div>
            <div style="font-size:13px;color:#888880">Upload an image to get started</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        k_values = sorted({k1, k2, k3})

        with st.spinner("Computing SVD..."):
            file_bytes = uploaded.getvalue()
            A, U, S, Vt = load_and_svd(file_bytes)
            m, n = A.shape
            cum_energy = np.cumsum(S ** 2) / np.sum(S ** 2) * 100

            results = {}
            for k in k_values:
                A_k   = rank_k_approx(U, S, Vt, k)
                error = frobenius_error(A, A_k)
                pct, orig, comp = storage_savings(m, n, k)
                energy = cum_energy[min(k, len(S)) - 1]
                results[k] = {
                    "matrix":  np.clip(A_k, 0, 255).astype(np.uint8),
                    "error":   error,
                    "savings": pct,
                    "energy":  energy,
                    "orig":    orig,
                    "comp":    comp,
                }

        # ── Top metric cards ──────────────────────────────────────────────────
        mc = st.columns(len(k_values))
        for col, k in zip(mc, k_values):
            r = results[k]
            with col:
                st.markdown(f"""
                <div class="card-sm">
                    <div class="metric-label">Rank k = {k}</div>
                    <div class="metric-value">{r['savings']:.1f}<span class="metric-unit">% saved</span></div>
                    <div class="metric-sub">Error: {r['error']:.1f} &nbsp;|&nbsp; Energy: {r['energy']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Comparison plot ───────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)

        n_panels = 1 + len(k_values)
        fig1, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.8))
        fig1.patch.set_facecolor("#ffffff")
        for ax in axes:
            ax.set_facecolor("#ffffff")

        axes[0].imshow(A, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title(f"Original\n{m} x {n}", fontsize=9, fontweight="bold", color="#1a1a18")
        axes[0].axis("off")

        for ax, k in zip(axes[1:], k_values):
            r = results[k]
            ax.imshow(r["matrix"], cmap="gray", vmin=0, vmax=255)
            ax.set_title(
                f"Rank-{k}\n{r['savings']:.1f}% saved  |  err {r['error']:.0f}",
                fontsize=9, color="#1a1a18"
            )
            ax.axis("off")

        plt.tight_layout(pad=1.2)
        st.pyplot(fig1, use_container_width=True)
        compression_bytes = fig_to_bytes(fig1)
        plt.close(fig1)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Spectrum plot ─────────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Singular Value Spectrum</div>', unsafe_allow_html=True)

        display_n = min(200, len(S))
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
        fig2.patch.set_facecolor("#ffffff")
        for ax in (ax1, ax2):
            ax.set_facecolor("#fafaf8")
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["left","bottom"]].set_color("#e0e0da")
            ax.tick_params(colors="#888880", labelsize=9)

        ax1.semilogy(S[:display_n], color="#1a9e8f", linewidth=1.8)
        ax1.set_xlabel("Index  i", fontsize=9, color="#888880")
        ax1.set_ylabel("Singular value (log)", fontsize=9, color="#888880")
        ax1.set_title("Singular Value Magnitude", fontsize=10, fontweight="bold", color="#1a1a18", pad=10)
        ax1.grid(True, alpha=0.25, color="#e0e0da")
        for k in k_values:
            if k <= display_n:
                ax1.axvline(x=k, linestyle="--", color="#f08040", alpha=0.8, linewidth=1.2, label=f"k={k}")
        ax1.legend(fontsize=8, framealpha=0.5)

        ax2.plot(cum_energy[:display_n], color="#1a9e8f", linewidth=1.8)
        ax2.set_xlabel("Number of components  k", fontsize=9, color="#888880")
        ax2.set_ylabel("Cumulative energy (%)", fontsize=9, color="#888880")
        ax2.set_title("Cumulative Energy Captured", fontsize=10, fontweight="bold", color="#1a1a18", pad=10)
        ax2.grid(True, alpha=0.25, color="#e0e0da")
        ax2.set_ylim(0, 101)
        for k in k_values:
            if k <= display_n:
                ax2.axvline(x=k, linestyle="--", color="#f08040", alpha=0.7, linewidth=1.2)
                ax2.axhline(y=cum_energy[min(k,len(S))-1], linestyle=":", color="#f08040", alpha=0.5, linewidth=1)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig2, use_container_width=True)
        spectrum_bytes = fig_to_bytes(fig2)
        plt.close(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Stats detail table ────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Compression Details</div>', unsafe_allow_html=True)

        header_c = st.columns([1, 1.5, 1.5, 1.5, 1.5])
        header_c[0].markdown('<div style="font-size:11px;font-weight:700;color:#888880;text-transform:uppercase;letter-spacing:.06em">Rank k</div>', unsafe_allow_html=True)
        header_c[1].markdown('<div style="font-size:11px;font-weight:700;color:#888880;text-transform:uppercase;letter-spacing:.06em">Frobenius Error</div>', unsafe_allow_html=True)
        header_c[2].markdown('<div style="font-size:11px;font-weight:700;color:#888880;text-transform:uppercase;letter-spacing:.06em">Storage Saved</div>', unsafe_allow_html=True)
        header_c[3].markdown('<div style="font-size:11px;font-weight:700;color:#888880;text-transform:uppercase;letter-spacing:.06em">Energy Captured</div>', unsafe_allow_html=True)
        header_c[4].markdown('<div style="font-size:11px;font-weight:700;color:#888880;text-transform:uppercase;letter-spacing:.06em">Values Stored</div>', unsafe_allow_html=True)

        st.markdown('<hr style="border:none;border-top:1px solid #e8e8e4;margin:8px 0">', unsafe_allow_html=True)

        for k in k_values:
            r = results[k]
            row = st.columns([1, 1.5, 1.5, 1.5, 1.5])
            row[0].markdown(f'<div style="font-size:14px;font-weight:700;color:#1a1a18">k = {k}</div>', unsafe_allow_html=True)
            row[1].markdown(f'<div style="font-size:14px;color:#1a1a18">{r["error"]:.2f}</div>', unsafe_allow_html=True)
            row[2].markdown(f'<div style="font-size:14px;font-weight:700;color:#1a9e8f">{r["savings"]:.2f}%</div>', unsafe_allow_html=True)
            row[3].markdown(f'<div style="font-size:14px;color:#1a1a18">{r["energy"]:.2f}%</div>', unsafe_allow_html=True)
            row[4].markdown(f'<div style="font-size:13px;color:#888880">{r["orig"]:,} → {r["comp"]:,}</div>', unsafe_allow_html=True)
            st.markdown('<hr style="border:none;border-top:1px solid #f0efeb;margin:6px 0">', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Download buttons ──────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Download Results</div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2, gap="medium")
        with dl1:
            st.download_button(
                "Download Comparison Plot",
                data=compression_bytes,
                file_name="compression_results.png",
                mime="image/png",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "Download Spectrum Plot",
                data=spectrum_bytes,
                file_name="sv_spectrum.png",
                mime="image/png",
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

#!/usr/bin/env python3
"""
Acoustic Epidemiology: Publication-Quality Figure Generation Script
Generates publication-ready figures for IEEE manuscript:
- Fig 1: Overall System Architecture
- Fig 2: Audio Preprocessing Pipeline (Waveform & Mel-Spectrogram)
- Fig 3: CNN Architecture with Dual Pooling
- Fig 4: Patient-Disjoint Dataset Partitioning
- Fig 5: Confusion Matrices (Baseline Center-Crop vs. Sliding-Window)
- Fig 6: Sliding-Window Multi-Scale Inference Mechanics
- Fig 7: Class-Wise Performance Comparison Bar Chart
"""

import os
import sys
from pathlib import Path
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "docs" / "research_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling for IEEE publications
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def generate_fig4_partitioning():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")

    # Draw Boxes
    def draw_box(x, y, w, h, title, text, color="#EBF3FB", edgecolor="#2B5B84"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                      facecolor=color, edgecolor=edgecolor, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.08, title, ha="center", va="top", weight="bold", fontsize=10, color="#1A365D")
        ax.text(x + w/2, y + 0.08, text, ha="center", va="bottom", fontsize=8.5, color="#2D3748")

    # Main Dataset
    draw_box(0.05, 0.72, 0.90, 0.22, "SPRSound Master Database (2,683 Total Manifest Recordings, 356 Patients)",
             "70 Poor Quality Excluded (Motion/Crying/SNR)  |  2,613 Valid Auscultation Recordings Evaluated\nStandardized: 8,000 Hz, 16-bit Mono PCM, 4 Thoracic Auscultation Sites (p1, p2, p3, p4)",
             color="#EDF2F7", edgecolor="#4A5568")

    # Splits
    draw_box(0.05, 0.35, 0.58, 0.26, "Cross-Validation Pool (train_cv)",
             "1,826 Total Records  |  251 Disjoint Patients\n"
             "  * Training Folds 1–4: 1,418 Valid Records (202 Patients)\n"
             "  * Validation Fold 0: 354 Valid Records (49 Patients)\n"
             "Zero Patient Leakage Across Folds (StratifiedGroupKFold)",
             color="#EBF8FF", edgecolor="#3182CE")

    draw_box(0.67, 0.35, 0.28, 0.26, "Independent Held-Out Test",
             "held_out_inter_test\n347 Valid Records\n41 Unseen Patients\nSTRICTLY 0 PATIENT OVERLAP",
             color="#FEFCBF", edgecolor="#D69E2E")

    draw_box(0.05, 0.05, 0.43, 0.20, "Intra-Subject Benchmark",
             "intra_subject_test: 136 Records\n13 Shared Patients (Segregated)",
             color="#F7FAFC", edgecolor="#A0AEC0")

    draw_box(0.52, 0.05, 0.43, 0.20, "Secondary 2023 Benchmark",
             "test_2023: 358 Valid Records\n51 Patients (Zero 2022 Overlap)",
             color="#F7FAFC", edgecolor="#A0AEC0")

    # Connectors
    ax.annotate("", xy=(0.34, 0.62), xytext=(0.34, 0.72), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2B5B84"))
    ax.annotate("", xy=(0.81, 0.62), xytext=(0.81, 0.72), arrowprops=dict(arrowstyle="->", lw=1.5, color="#B7791F"))

    plt.title("Patient-Disjoint Dataset Partitioning Architecture", pad=10, weight="bold", fontsize=11)
    fig.savefig(OUTPUT_DIR / "fig4_patient_partitioning.png")
    plt.close()
    print("Saved fig4_patient_partitioning.png")


def generate_fig5_confusion_matrices():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    classes = ["Normal", "CAS", "DAS", "CAS & DAS"]

    # Baseline Center-Crop
    cm_base = np.array([
        [187,   0,  49,   5],
        [ 31,   1,  22,  11],
        [  7,   1,  11,   5],
        [  4,   0,   9,   4]
    ])

    # Sliding-Window Abnormality-Max (Stride 64)
    cm_sw = np.array([
        [179,   0,  57,   5],
        [ 31,   0,  25,   9],
        [  7,   0,  12,   5],
        [  4,   0,  11,   2]
    ])

    im1 = ax1.imshow(cm_base, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=200)
    ax1.set_title("(a) Baseline Center-Crop ($N=347$)\nAcc: 58.50% | BAcc: 37.12%", fontsize=10, weight="bold")
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(classes, rotation=25)
    ax1.set_yticklabels(classes)
    ax1.set_ylabel("True Class", weight="bold")
    ax1.set_xlabel("Predicted Class", weight="bold")

    for i in range(4):
        for j in range(4):
            val = cm_base[i, j]
            color = "white" if val > 90 else "black"
            ax1.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=10, weight="bold")

    im2 = ax2.imshow(cm_sw, interpolation='nearest', cmap=plt.cm.Greens, vmin=0, vmax=200)
    ax2.set_title("(b) Sliding-Window Abnormality-Max ($S=64$)\nAcc: 55.62% | BAcc: 34.01%", fontsize=10, weight="bold")
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(classes, rotation=25)
    ax2.set_yticklabels(classes)
    ax2.set_xlabel("Predicted Class", weight="bold")

    for i in range(4):
        for j in range(4):
            val = cm_sw[i, j]
            color = "white" if val > 90 else "black"
            ax2.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=10, weight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_confusion_matrices.png")
    plt.close()
    print("Saved fig5_confusion_matrices.png")


def generate_fig7_performance_comparison():
    fig, ax = plt.subplots(figsize=(8, 4.2))

    categories = ["Normal", "CAS (Wheezes)", "DAS (Crackles)", "CAS & DAS", "Balanced Accuracy"]
    
    # Values in %
    cc_vals = [77.59, 1.54, 45.83, 23.53, 37.12]
    sw_mean_vals = [82.16, 0.00, 41.67, 5.88, 32.43]
    sw_abn_vals = [74.27, 0.00, 50.00, 11.76, 34.01]
    sw_32_vals = [72.61, 0.00, 54.17, 17.65, 36.11]

    x = np.arange(len(categories))
    width = 0.20

    rects1 = ax.bar(x - 1.5*width, cc_vals, width, label="Center-Crop Baseline", color="#3182CE", edgecolor="black")
    rects2 = ax.bar(x - 0.5*width, sw_mean_vals, width, label="SW Stride-64 Mean", color="#A0AEC0", edgecolor="black")
    rects3 = ax.bar(x + 0.5*width, sw_abn_vals, width, label="SW Stride-64 AbnMax (Locked)", color="#38A169", edgecolor="black")
    rects4 = ax.bar(x + 1.5*width, sw_32_vals, width, label="SW Stride-32 AbnMax (Exploratory)", color="#D69E2E", edgecolor="black")

    ax.set_ylabel("Recall / Balanced Accuracy (%)", weight="bold")
    ax.set_title("Performance Comparison on Independent Held-Out Benchmark ($N=347$)", weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, weight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95)

    # Highlight crackle gain
    ax.annotate("+4.17% DAS Gain", xy=(2 + 0.5*width, 50.0), xytext=(2 + 0.5*width, 68),
                arrowprops=dict(facecolor="#38A169", shrink=0.08, width=1.5, headwidth=6),
                ha="center", fontsize=8.5, weight="bold", color="#22543D")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_performance_comparison.png")
    plt.close()
    print("Saved fig7_performance_comparison.png")


def generate_fig2_spectrogram():
    # Load actual spectrogram from disk
    records_dir = PROJECT_ROOT / "ml" / "features" / "extracted" / "records"
    sample_npz = list(records_dir.glob("held_out_inter_test_*_record.npz"))[0]
    data = np.load(sample_npz)
    spec = data["spectrogram"]  # shape (128, T)
    T = spec.shape[1]
    duration = float(data.get("raw_duration_seconds", T * 0.064))

    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma", extent=[0, duration, 50, 2000])
    ax.set_title(f"Log Mel-Spectrogram: Record {data['record_id']} ({data['label']}, Duration: {duration:.2f}s)", fontsize=10, weight="bold")
    ax.set_xlabel("Time (seconds)", weight="bold")
    ax.set_ylabel("Frequency (Hz)", weight="bold")

    # Indicate Center Crop vs Sliding Window
    center_start = max(0, (duration - 8.192) / 2)
    rect = patches.Rectangle((center_start, 50), 8.192, 1950, linewidth=2, edgecolor="#00FFAA", facecolor="none", linestyle="--", label="8.192s Center-Crop Window")
    ax.add_patch(rect)
    ax.legend(loc="upper right", framealpha=0.9)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Normalized Energy", weight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_preprocessing_pipeline.png")
    plt.close()
    print("Saved fig2_preprocessing_pipeline.png")


def main():
    generate_fig4_partitioning()
    generate_fig5_confusion_matrices()
    generate_fig7_performance_comparison()
    generate_fig2_spectrogram()
    print("All publication figures successfully created in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

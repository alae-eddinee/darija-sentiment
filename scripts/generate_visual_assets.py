"""
Generate publication-quality, dark-themed visuals for Darija Sentiment Classifier.
Outputs:
  - results/visual_benchmark_comparison.png
  - results/visual_dataset_architecture.png
  - results/visual_arabizi_phoneme_guide.png
"""

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

AR_FONT = FontProperties(fname="/System/Library/Fonts/GeezaPro.ttc", size=18, weight="bold")

# Styling defaults
DARK_BG = "#09090b"
CARD_BG = "#18181b"
TEXT_WHITE = "#f4f4f5"
TEXT_MUTED = "#a1a1aa"
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#ef4444"
ACCENT_SLATE = "#71717a"
ACCENT_EMERALD = "#059669"
BORDER_COLOR = "#27272a"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "Beirut", "Geeza Pro", "Arial Unicode MS", "DejaVu Sans"]


def generate_benchmark_chart(output_path: Path):
    """Generate high-contrast benchmark comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    slices = ["Overall (7.6k)", "Arabic Script (5.7k)", "Arabizi Script (1.7k)", "Mixed Script (0.2k)"]
    before_f1 = [0.7436, 0.7310, 0.6678, 0.8564]
    after_f1 = [0.7539, 0.7320, 0.8924, 0.7485]
    
    y = np.arange(len(slices))
    height = 0.32

    # Draw bars
    bars1 = ax.barh(y - height/2, before_f1, height, label="Initial Model (36k Dataset)", color="#3f3f46", edgecolor="none")
    bars2 = ax.barh(y + height/2, after_f1, height, label="Current Model (51k Dataset + Blessing Inoculation)", color=ACCENT_GREEN, edgecolor="none")

    # Add data labels
    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 0.015, bar.get_y() + bar.get_height()/2, f"{w:.3f}", 
                va="center", ha="left", color=TEXT_MUTED, fontsize=10, fontweight="600")

    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 0.015, bar.get_y() + bar.get_height()/2, f"{w:.3f}", 
                va="center", ha="left", color=TEXT_WHITE, fontsize=10, fontweight="bold")

    # Highlight Arabizi leap
    ax.annotate("+22.5% Macro-F1\n(94.1% Accuracy)", 
                xy=(0.8924, 2 + height/2), 
                xytext=(0.70, 2.7),
                arrowprops=dict(facecolor=ACCENT_GREEN, edgecolor=ACCENT_GREEN, shrink=0.08, width=1.5, headwidth=6),
                color=ACCENT_GREEN, fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#064e3b", ec=ACCENT_GREEN, lw=1))

    # Styling
    ax.set_xlim(0, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(slices, color=TEXT_WHITE, fontsize=11, fontweight="600")
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 Score", color=TEXT_MUTED, fontsize=11, labelpad=10)
    
    # Hide spines
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.xaxis.grid(True, color="#27272a", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_MUTED, labelsize=10, left=False)

    # Title
    fig.suptitle("Moroccan Darija — Model Performance by Script", color=TEXT_WHITE, fontsize=16, fontweight="bold", y=0.98)
    ax.set_title("Held-out evaluation comparing initial 36k baseline vs. expanded 51k corpus", color=TEXT_MUTED, fontsize=10, pad=14)

    # Legend
    legend = ax.legend(loc="lower right", facecolor=CARD_BG, edgecolor=BORDER_COLOR, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(TEXT_WHITE)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor=DARK_BG, edgecolor="none")
    plt.close()
    print(f"✅ Generated: {output_path}")


def generate_dataset_architecture(output_path: Path):
    """Generate multi-panel dataset distribution infographic."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.6), dpi=300)
    fig.patch.set_facecolor(DARK_BG)

    # 1. Script Donut
    scripts = ["Arabic\n(38.1k)", "Arabizi\n(11.6k)", "Mixed\n(1.3k)"]
    script_counts = [38104, 11635, 1315]
    colors_script = ["#0284c7", "#8b5cf6", "#f97316"]
    
    wedges, texts, autotexts = ax1.pie(
        script_counts, labels=scripts, autopct="%1.1f%%", pctdistance=0.75,
        colors=colors_script, startangle=140,
        wedgeprops=dict(width=0.45, edgecolor=DARK_BG, linewidth=2)
    )
    for t in texts:
        t.set_color(TEXT_WHITE)
        t.set_fontsize(9)
        t.set_fontweight("600")
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax1.set_title("Scripts Breakdown\n51,054 Total Samples", color=TEXT_WHITE, fontsize=12, fontweight="bold", pad=10)

    # 2. Source Breakdown
    sources = ["MYC\n(YouTube)", "MAC\n(Twitter)", "TSAC\n(Dialect)", "MSDA\n(UM6P)", "Curated\n(Blessings)", "MSAC\n(E-Comm)"]
    source_counts = [18699, 12337, 10668, 5165, 2211, 1974]
    
    ax2.set_facecolor(DARK_BG)
    y_pos = np.arange(len(sources))
    bars = ax2.barh(y_pos, source_counts, color="#3b82f6", edgecolor="none", height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sources, color=TEXT_WHITE, fontsize=9)
    ax2.invert_yaxis()
    for spine in ["top", "right", "left", "bottom"]:
        ax2.spines[spine].set_visible(False)
    ax2.xaxis.grid(True, color="#27272a", linestyle="--", alpha=0.6)
    ax2.tick_params(colors=TEXT_MUTED, labelsize=9, left=False)
    ax2.set_title("Corpus Sources\nMulti-Domain Aggregation", color=TEXT_WHITE, fontsize=12, fontweight="bold", pad=10)
    for b in bars:
        w = b.get_width()
        ax2.text(w + 300, b.get_y() + b.get_height()/2, f"{w:,}", va="center", color=TEXT_MUTED, fontsize=8.5)

    # 3. Sentiment Balance
    classes = ["Positive", "Negative", "Neutral"]
    class_counts = [25627, 20094, 5333]
    colors_class = [ACCENT_GREEN, ACCENT_RED, "#64748b"]

    ax3.set_facecolor(DARK_BG)
    x_pos = np.arange(len(classes))
    bars3 = ax3.bar(x_pos, class_counts, color=colors_class, edgecolor="none", width=0.55)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(classes, color=TEXT_WHITE, fontsize=10, fontweight="600")
    for spine in ["top", "right", "left", "bottom"]:
        ax3.spines[spine].set_visible(False)
    ax3.yaxis.grid(True, color="#27272a", linestyle="--", alpha=0.6)
    ax3.tick_params(colors=TEXT_MUTED, labelsize=9, bottom=False)
    ax3.set_title("Sentiment Polarity\n3-Class Balance", color=TEXT_WHITE, fontsize=12, fontweight="bold", pad=10)
    for b in bars3:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2, h + 500, f"{h/1000:.1f}k\n({h/51054*100:.0f}%)", ha="center", color=TEXT_WHITE, fontsize=9)

    fig.suptitle("Moroccan Darija Dataset Composition (51,054 Comments)", color=TEXT_WHITE, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor=DARK_BG, edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"✅ Generated: {output_path}")


def generate_arabizi_guide(output_path: Path):
    """Generate Arabizi numeral decoder graphic."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    items = [
        ("3", "ع", "Ain", "3afak (please), wa3er (awesome), 3lach (why)", "#8b5cf6"),
        ("7", "ح", "Haa", "7chouma (shame), se7a (health), 7bibi (friend)", "#10b981"),
        ("9", "ق", "Qaf", "9rib (near), 9ahwa (coffee), 9bel (before)", "#ef4444"),
        ("5", "خ", "Khaa", "5oya (brother), 5ir (good), 5edma (work)", "#f59e0b"),
        ("2", "ء", "Hamza", "so2al (question), ra2i (opinion), mas2oul", "#06b6d4"),
    ]

    fig.suptitle("Moroccan Arabizi Phoneme Decoder", color=TEXT_WHITE, fontsize=16, fontweight="bold", y=0.96)
    ax.text(0.5, 0.88, "How numbers substitute Arabic sounds missing in standard Latin keyboards", 
            ha="center", color=TEXT_MUTED, fontsize=10, transform=ax.transAxes)

    y_start = 0.72
    y_step = 0.15

    for idx, (digit, arabic, name, examples, col) in enumerate(items):
        y = y_start - idx * y_step
        
        # Pill background
        rect = patches.FancyBboxPatch((0.02, y - 0.05), 0.96, 0.12, boxstyle="round,pad=0.01,rounding_size=0.02",
                                      facecolor=CARD_BG, edgecolor=BORDER_COLOR, lw=1, transform=ax.transAxes)
        ax.add_patch(rect)

        # Digit badge
        ax.text(0.06, y + 0.01, digit, color=col, fontsize=18, fontweight="bold", transform=ax.transAxes)
        ax.text(0.12, y + 0.01, "->", color=TEXT_MUTED, fontsize=14, fontweight="bold", transform=ax.transAxes)
        ax.text(0.18, y + 0.005, arabic, color=TEXT_WHITE, fontproperties=AR_FONT, transform=ax.transAxes)
        ax.text(0.25, y + 0.01, f"({name})", color=col, fontsize=11, fontweight="600", transform=ax.transAxes)
        ax.text(0.38, y + 0.01, examples, color=TEXT_MUTED, fontsize=10, transform=ax.transAxes)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor=DARK_BG, edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"✅ Generated: {output_path}")


if __name__ == "__main__":
    res_dir = Path("results")
    generate_benchmark_chart(res_dir / "visual_benchmark_comparison.png")
    generate_dataset_architecture(res_dir / "visual_dataset_architecture.png")
    generate_arabizi_guide(res_dir / "visual_arabizi_phoneme_guide.png")

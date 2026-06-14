import os
import matplotlib.pyplot as plt
import numpy as np

def plot_pareto():
    # Setup styles
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Data points: (Latency in ms, Success Rate in %, Label, Color, Marker)
    data = [
        (164.0, 76.5, "OpenVLA (7B)", "#d62728", "D"),
        (138.0, 85.8, "π0.5 (VLA)", "#ff7f0e", "X"),
        (52.0, 96.9, "FLOWER (Florence-2)", "#9467bd", "s"),
        (22.0, 97.3, "ThinkProprio (Florence-2 + Prune)", "#1f77b4", "^"),
        (40.0, 95.0, "Native ACT (No compression)", "#7f7f7f", "o"),
        (22.0, 75.0, "Hard Selection (85% Pruning, Ours)", "#bcbd22", "v"),
        (26.0, 98.2, "ProMerge (ToMe Merging, Ours)", "#2ca02c", "*")
    ]

    for latency, sr, label, color, marker in data:
        size = 180 if "Ours" in label else 120
        ax.scatter(latency, sr, label=label, color=color, marker=marker, s=size, edgecolors='black', zorder=5)
        
        # Adjust text offset dynamically to avoid overlap
        x_offset = 6 if latency < 100 else -40
        y_offset = -3 if label == "ThinkProprio (Florence-2 + Prune)" else 2
        if label == "ProMerge (ToMe Merging, Ours)":
            y_offset = 4
            x_offset = 8
        if label == "Hard Selection (85% Pruning, Ours)":
            y_offset = -12
            x_offset = 6
        ax.annotate(label, (latency, sr), textcoords="offset points", xytext=(x_offset, y_offset), 
                    fontsize=9, fontweight='bold' if "Ours" in label else 'normal', alpha=0.9)

    # Drawing Pareto Frontier curves for visual interpretation
    # Front is top-left (low latency, high success rate)
    ax.plot([22.0, 26.0, 52.0, 138.0, 164.0], [97.3, 98.2, 96.9, 85.8, 76.5], 
            color='#2ca02c', linestyle='--', linewidth=2, alpha=0.5, label='Pareto Frontier (ToMe Merged)')

    # Labels and Titles
    ax.set_title("Pareto Frontier: Control Latency vs. Task Success Rate (LIBERO / Occlusion)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("End-to-End Control Latency (ms) [Lower is Better]", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Average Task Success Rate (%) [Higher is Better]", fontsize=11, fontweight='semibold')
    
    ax.set_xlim(0, 180)
    ax.set_ylim(60, 105)

    # Grid styling
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Legend
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none', fontsize=10)

    plt.tight_layout()
    
    # Save path in brain artifacts and project root
    artifacts_dir = "/Users/jiucai/.gemini/antigravity-ide/brain/45d1f158-cffc-4176-9c13-283ec5553bfe"
    os.makedirs(artifacts_dir, exist_ok=True)
    fig.savefig(os.path.join(artifacts_dir, "pareto_frontier.png"), bbox_inches='tight')
    fig.savefig("pareto_frontier.png", bbox_inches='tight')
    plt.close()
    print("✨ Pareto Frontier chart successfully saved to pareto_frontier.png!")

if __name__ == "__main__":
    plot_pareto()

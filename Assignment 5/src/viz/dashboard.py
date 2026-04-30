"""Static training dashboards (multi-panel)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training(csv_paths: dict, out_dir: str | Path, fname: str = "training_dashboard.png") -> Path:
    """csv_paths: {agent_name: path_to_csv}. Returns path to saved figure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name, p in csv_paths.items():
        if not Path(p).exists():
            continue
        d = pd.read_csv(p)
        if len(d) == 0:
            continue
        axes[0, 0].plot(d["episode"], d["return"], alpha=0.3, label=f"{name} (raw)")
        axes[0, 0].plot(d["episode"], d["return"].rolling(100, min_periods=1).mean(),
                        label=f"{name} (avg100)")
        axes[0, 1].plot(d["episode"], d["max_tile"], alpha=0.5, label=name)
        axes[1, 0].plot(d["episode"], d["return"].rolling(100, min_periods=1).mean(), label=name)
        if "loss_or_td_error" in d.columns:
            loss = pd.to_numeric(d["loss_or_td_error"], errors="coerce")
            axes[1, 1].plot(d["episode"], loss.rolling(100, min_periods=1).mean(), label=name)
    axes[0, 0].set_title("Return per episode")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[0, 1].set_title("Max tile per episode")
    axes[0, 1].set_yscale("log", base=2)
    axes[0, 1].legend()
    axes[1, 0].set_title("Rolling-100 return")
    axes[1, 0].legend()
    axes[1, 1].set_title("Loss / TD error (smoothed)")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.set_xlabel("episode")
    fig.tight_layout()
    out = out_dir / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_tile_distribution(eval_paths: dict, out_dir: str | Path,
                           fname: str = "tile_distribution.png") -> Path:
    """Bar chart of highest-tile distribution from eval JSON files."""
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, p in eval_paths.items():
        if not Path(p).exists():
            continue
        m = json.loads(Path(p).read_text())
        for tile, prob in m["pct_reaching"].items():
            rows.append({"agent": name, "tile": int(tile), "prob": prob})
    sns.set_theme()
    df = pd.DataFrame(rows).sort_values(["agent", "tile"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="tile", y="prob", hue="agent", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Probability of reaching tile T")
    fig.tight_layout()
    out = out_dir / fname
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out

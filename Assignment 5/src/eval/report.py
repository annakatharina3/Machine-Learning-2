"""Build comparison artifacts (CSV + figures) used in the writeup."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def build_comparison(eval_paths: dict, train_logs: dict, out_dir: str | Path) -> pd.DataFrame:
    """eval_paths: {agent_name: path_to_eval.json}
       train_logs: {agent_name: path_to_csv}
       out_dir: where to write comparison.csv and figures.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, p in eval_paths.items():
        if not Path(p).exists():
            continue
        m = json.loads(Path(p).read_text())
        row = {
            "agent": name,
            "mean_score": m["mean_score"],
            "median_score": m["median_score"],
            "max_score": m["max_score"],
            "mean_max_tile": m["mean_max_tile"],
            "max_max_tile": m["max_max_tile"],
            "mean_steps": m["mean_steps"],
        }
        for t, v in m["pct_reaching"].items():
            row[f"pct_>={t}"] = v
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("mean_score", ascending=False)
    df.to_csv(out_dir / "comparison.csv", index=False)

    sns.set_theme()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="agent", y="mean_score", ax=ax)
    ax.set_title("Mean episode score (eval)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_scores.png", dpi=130)
    plt.close(fig)

    long_rows = []
    for _, r in df.iterrows():
        for t in (512, 1024, 2048, 4096, 8192):
            col = f"pct_>={t}"
            if col in r:
                long_rows.append({"agent": r["agent"], "tile": str(t), "prob": r[col]})
    long_df = pd.DataFrame(long_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=long_df, x="tile", y="prob", hue="agent", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Probability of reaching tile T")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_tile_reach.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, log in train_logs.items():
        if not Path(log).exists():
            continue
        d = pd.read_csv(log)
        if "return" not in d.columns or len(d) == 0:
            continue
        roll = d["return"].rolling(100, min_periods=1).mean()
        ax.plot(d["episode"], roll, label=name)
    ax.set_xlabel("episode")
    ax.set_ylabel("rolling-100 return")
    ax.set_title("Training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_training_curves.png", dpi=130)
    plt.close(fig)

    return df

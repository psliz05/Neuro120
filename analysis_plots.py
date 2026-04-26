import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data_17dogs_1.csv", sep=";", decimal=",")
df_long = pd.read_csv("data_17dogs_long_1.csv", sep=";", decimal=",")


# ── Helper: Cohen's d (paired, within-subject) ────────────────────────────────
def cohens_d_paired(a, b):
    diff = a - b
    return diff.mean() / diff.std(ddof=1)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — Per-Dog Species NDI (d') in the 250–650 ms window
# ══════════════════════════════════════════════════════════════════════════════

window_col = "250-650"
electrode = "Fz"  # choose Fz or average Fz+Cz — adjust if needed

records = []
for dog_id in sorted(df["ID"].unique()):
    sub = df[(df["ID"] == dog_id) & (df["electrode"] == electrode)]
    dog_vals = sub[sub["species"] == "dog"][window_col].values
    human_vals = sub[sub["species"] == "human"][window_col].values
    if len(dog_vals) > 0 and len(human_vals) > 0:
        # NDI: dog minus human → positive = more response to dog
        d = cohens_d_paired(pd.Series(dog_vals), pd.Series(human_vals))
        records.append({"ID": dog_id, "d": d})

ndi_df = pd.DataFrame(records)
mean_d = ndi_df["d"].mean()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar colours: blue = positive, red = negative
colors = ["steelblue" if v >= 0 else "firebrick" for v in ndi_df["d"]]
axes[0].bar(ndi_df["ID"], ndi_df["d"], color=colors, edgecolor="white", linewidth=0.5)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].axhline(
    mean_d,
    color="steelblue",
    linewidth=1.5,
    linestyle="--",
    label=f"Mean d' = {mean_d:.2f}",
)
axes[0].set_xlabel("Dog ID")
axes[0].set_ylabel("d' (species discriminability)")
axes[0].set_title("Per-Dog Species NDI\n250–650ms window")
axes[0].set_xticks(ndi_df["ID"])
axes[0].legend(loc="upper right")

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — Discrimination Metric over time (Species & Valence)
# ══════════════════════════════════════════════════════════════════════════════

# Sliding-window columns from the long file (100-ms steps)
time_cols = [
    c
    for c in df_long.columns
    if "-" in c and c != "base" and not any(x in c for x in ["species", "valence", "electrode", "ID"])
]


# Parse centre of each window in ms
def window_centre(col):
    lo, hi = col.split("-")
    return (int(lo) + int(hi)) / 2


times = [window_centre(c) for c in time_cols]

species_d = []
valence_d = []

for col in time_cols:
    sub = df_long[df_long["electrode"] == electrode]

    # Species effect: dog vs human (collapsed over valence)
    dog_ = sub[sub["species"] == "dog"][col]
    human_ = sub[sub["species"] == "human"][col]
    # Use independent-samples d since different dogs contribute to each level
    pooled_sd = np.sqrt((dog_.std(ddof=1) ** 2 + human_.std(ddof=1) ** 2) / 2)
    species_d.append((dog_.mean() - human_.mean()) / pooled_sd if pooled_sd else 0)

    # Valence effect: positive vs neutral (collapsed over species)
    pos_ = sub[sub["valence"] == "positive"][col]
    neut_ = sub[sub["valence"] == "neutral"][col]
    pooled_sd = np.sqrt((pos_.std(ddof=1) ** 2 + neut_.std(ddof=1) ** 2) / 2)
    valence_d.append((pos_.mean() - neut_.mean()) / pooled_sd if pooled_sd else 0)

axes[1].plot(times, species_d, color="steelblue", linewidth=2, label="Species")
axes[1].plot(times, valence_d, color="firebrick", linewidth=2, linestyle="--", label="Valence")
axes[1].axhline(0, color="grey", linewidth=0.8, linestyle=":")
axes[1].set_xlabel("Time after stimulus onset (ms)")
axes[1].set_ylabel("Discrimination Metric")
axes[1].set_title("Discrimination Metric")
axes[1].legend()

plt.tight_layout()
plt.savefig("cohens_d_graphs.png", dpi=150)
plt.show()

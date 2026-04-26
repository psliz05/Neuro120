import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('data_17dogs_long_1.csv', sep=';', decimal=',')

electrode = 'Fz'
sub = df[df['electrode'] == electrode]

# ── Get sliding-window columns and their centre times ─────────────────────────
skip = {'ID', 'species', 'valence', 'electrode', 'base'}
time_cols = [c for c in df.columns if c not in skip]


def window_centre(col):
    lo, hi = col.split('-')
    return (int(lo) + int(hi)) / 2


times = [window_centre(c) for c in time_cols]

# ── Grand-average per species across all valences and dogs ────────────────────
human_means = sub[sub['species'] == 'human'][time_cols].mean()
dog_means = sub[sub['species'] == 'dog'][time_cols].mean()

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(times, human_means, color='steelblue', linewidth=2, label='Human vocalizations')
ax.plot(times, dog_means, color='orange', linewidth=2, label='Dog vocalizations')

# Stimulus onset line and zero amplitude line
ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.6)
ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)

ax.set_xlabel('Time after stimulus onset (ms)')
ax.set_ylabel('ERP amplitude (µV)')
ax.set_title('Dog brain response to human vs. dog vocalizations')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('erp_species_waveform.png', dpi=150)
plt.show()

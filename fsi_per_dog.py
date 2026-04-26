import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('data_17dogs_1.csv', sep=';', decimal=',')

window_col = '250-650'
electrode = 'Fz'

sub = df[df['electrode'] == electrode]

# ── Compute FSI per dog ───────────────────────────────────────────────────────
# FSI = (human_mean - dog_mean) / (|human_mean| + |dog_mean|)
# Ranges from -1 (all dog) to +1 (all human)

records = []
for dog_id in sorted(sub['ID'].unique()):
    dog_sub = sub[sub['ID'] == dog_id]
    human_mean = dog_sub[dog_sub['species'] == 'human'][window_col].mean()
    dog_mean = dog_sub[dog_sub['species'] == 'dog'][window_col].mean()
    denom = abs(human_mean) + abs(dog_mean)
    fsi = (human_mean - dog_mean) / denom if denom != 0 else 0
    records.append({'ID': dog_id, 'FSI': fsi})

fsi_df = pd.DataFrame(records).sort_values('FSI').reset_index(drop=True)
group_mean = fsi_df['FSI'].mean()

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

colors = ['steelblue' if v >= 0 else 'tomato' for v in fsi_df['FSI']]
y_pos = range(len(fsi_df))

ax.barh(list(y_pos), fsi_df['FSI'], color=colors, edgecolor='white', linewidth=0.4)

# Group mean dashed line
ax.axvline(
    group_mean,
    color='black',
    linewidth=1.5,
    linestyle='--',
    label=f'Group mean (FSI = {group_mean:.2f})',
)
ax.axvline(0, color='black', linewidth=0.6, alpha=0.4)

# Dog labels
ax.set_yticks(list(y_pos))
ax.set_yticklabels([f'Dog {int(i)}' for i in fsi_df['ID']])

ax.set_xlim(-1.0, 1.0)
ax.set_xlabel('Familiarity Selectivity Index (FSI)')
ax.set_title('Neural preference for human vocalizations in each dog')

# Custom legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='steelblue', label='Stronger response to human voices'),
    Patch(facecolor='tomato', label='Stronger response to dog voices'),
    plt.Line2D([0], [0], color='black', linestyle='--', label=f'Group mean (FSI = {group_mean:.2f})'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('fsi_per_dog.png', dpi=150)
plt.show()

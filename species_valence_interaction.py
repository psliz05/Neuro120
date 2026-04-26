import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('data_17dogs_1.csv', sep=';', decimal=',')

window_col = '250-650'
electrode = 'Fz'

sub = df[df['electrode'] == electrode]

# ── Compute cell means and SEs (per dog, then across dogs) ────────────────────
conditions = [
    ('human', 'positive'),
    ('human', 'neutral'),
    ('dog', 'positive'),
    ('dog', 'neutral'),
]

cell_stats = {}
for sp, val in conditions:
    vals = sub[(sub['species'] == sp) & (sub['valence'] == val)][window_col]
    cell_stats[(sp, val)] = {
        'mean': vals.mean(),
        'se': vals.sem(),
        'n': len(vals),
    }

# ── ANOVA (repeated-measures approximation via paired t / F) ──────────────────
# Pivot to per-dog wide format for each effect
dogs = sorted(sub['ID'].unique())

# Species effect: average over valence per dog
human_means = sub[sub['species'] == 'human'].groupby('ID')[window_col].mean()
dog_means = sub[sub['species'] == 'dog'].groupby('ID')[window_col].mean()

t_sp, p_sp = stats.ttest_rel(human_means, dog_means)
n = len(dogs)
F_sp = t_sp**2
df_sp = n - 1

# Valence effect
pos_means = sub[sub['valence'] == 'positive'].groupby('ID')[window_col].mean()
neut_means = sub[sub['valence'] == 'neutral'].groupby('ID')[window_col].mean()
t_val, p_val = stats.ttest_rel(pos_means, neut_means)
F_val = t_val**2

# Interaction: (human_pos - human_neut) vs (dog_pos - dog_neut)
human_diff = (
    sub[(sub['species'] == 'human') & (sub['valence'] == 'positive')].set_index('ID')[window_col]
    - sub[(sub['species'] == 'human') & (sub['valence'] == 'neutral')].set_index('ID')[window_col]
)
dog_diff = (
    sub[(sub['species'] == 'dog') & (sub['valence'] == 'positive')].set_index('ID')[window_col]
    - sub[(sub['species'] == 'dog') & (sub['valence'] == 'neutral')].set_index('ID')[window_col]
)
t_int, p_int = stats.ttest_rel(human_diff.reindex(dogs), dog_diff.reindex(dogs))
F_int = t_int**2


def sig_label(p):
    return '*' if p <= 0.05 else 'n.s.'


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

x = [0, 1]
xlabels = ['Positive', 'Neutral']

for sp, color, marker, ls, label in [
    ('human', 'steelblue', 'o', '-', 'Human vocalizations'),
    ('dog', 'firebrick', 's', '--', 'Dog vocalizations'),
]:
    means = [cell_stats[(sp, v)]['mean'] for v in ['positive', 'neutral']]
    ses = [cell_stats[(sp, v)]['se'] for v in ['positive', 'neutral']]
    ax.errorbar(
        x,
        means,
        yerr=ses,
        color=color,
        marker=marker,
        linestyle=ls,
        linewidth=2,
        markersize=8,
        capsize=6,
        label=label,
    )

ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_xlabel('Valence')
ax.set_ylabel('Mean ERP amplitude (µV)')
ax.set_title('Species × Valence Interaction\n250–650ms window (Fz electrode)')
ax.legend(loc='upper left')

# Stats box
stats_text = (
    f"Species:  F(1,{df_sp}) = {F_sp:.2f}, p = {p_sp:.3f} {sig_label(p_sp)}\n"
    f"Valence:  F(1,{df_sp}) = {F_val:.2f}, p = {p_val:.3f} {sig_label(p_val)}\n"
    f"Interact: F(1,{df_sp}) = {F_int:.2f}, p = {p_int:.3f} {sig_label(p_int)}"
)
ax.text(
    0.97,
    0.05,
    stats_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
)

plt.tight_layout()
plt.savefig('species_valence_interaction.png', dpi=150)
plt.show()

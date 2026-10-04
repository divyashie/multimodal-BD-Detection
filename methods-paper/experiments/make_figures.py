#!/usr/bin/env python3
"""
Generate the four figures for Paper 1 from the saved result files.

Reads results.json / section6.json; re-runs nothing. Writes PDF (for submission)
and PNG (for drafts) into methods-paper/figures/.

    python make_figures.py --as-run run_as_run --complete run_complete --out ../figures

Figure 1  Aligned vs within-class vs across-class permutation      [as_run]
Figure 2  OBF feature coverage by source file                      [structural]
Figure 3  Correction waterfall, fusion vs its own shuffle          [as_run]
Figure 4  Calibration error vs bin count, with bootstrap CIs       [as_run]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- house style: greyscale-safe, no chartjunk, serif to match the manuscript -------------
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'legend.frameon': False,
})
INK, MID, PALE, HILITE = '#222222', '#777777', '#cccccc', '#1f4e79'


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  wrote {name}.pdf / .png')


# ---------------------------------------------------------------- Figure 1
def figure1(R, out_dir):
    """The anchor: aligned and within-class shuffle are indistinguishable."""
    cv = R['A']['A2_cv']
    labels = ['Aligned\n(as reported)', 'Within-class\npermutation', 'Across-class\npermutation']
    keys = ['aligned', 'within_class_shuffle', 'across_class_shuffle']
    means = [cv[k][0] for k in keys]
    sds = [cv[k][1] for k in keys]

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    x = np.arange(3)
    ax.bar(x, means, yerr=sds, capsize=4, width=0.6,
           color=[HILITE, HILITE, PALE], edgecolor=INK, linewidth=0.8,
           error_kw=dict(ecolor=INK, lw=1))
    ax.axhline(1 / 3, ls=':', lw=1, color=MID)
    ax.text(2.45, 1 / 3 + 0.008, 'chance', ha='right', va='bottom', fontsize=7.5, color=MID)

    for xi, (m, s) in enumerate(zip(means, sds)):
        ax.text(xi, m + s + 0.015, f'{m:.3f}', ha='center', fontsize=8.5)

    # bracket over the two bars that matter
    y = max(means[0] + sds[0], means[1] + sds[1]) + 0.055
    ax.plot([0, 0, 1, 1], [y, y + 0.012, y + 0.012, y], lw=0.9, color=INK)
    ax.text(0.5, y + 0.02, 'identical to three decimals', ha='center', fontsize=8, style='italic')

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Macro-F1 (repeated 5-fold CV)')
    ax.set_ylim(0, 1.0)
    ax.set_title('Permuting within classes costs nothing', fontsize=10, loc='left', pad=10)
    save(fig, out_dir, 'fig1_within_class_permutation')


# ---------------------------------------------------------------- Figure 2
def figure2(out_dir):
    """Structural: no participant has both halves of the feature vector."""
    files = ['features.csv', 'adhd-info.csv', 'clinical-info.csv',
             'control-info.csv', 'depression-info.csv', 'schizophrenia-info.csv']
    feats = ['mean', 'sd', 'pctZeros', 'median', 'madrs', 'hads_d', 'asrs', 'mdq_pos']
    # coverage as audited in B2 (fraction of rows with the column present)
    cov = np.array([
        [1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00],   # features.csv
        [0.00, 0.00, 0.00, 0.00, 0.93, 0.91, 0.98, 0.96],   # adhd-info.csv
        [0.00, 0.00, 0.00, 0.00, 0.95, 0.92, 0.92, 0.90],   # clinical-info.csv
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],   # control-info.csv
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],   # depression-info.csv
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],   # schizophrenia-info.csv
    ])
    people = [77, 45, 40, 32, 23, 22]

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.imshow(cov, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            v = cov[i, j]
            ax.text(j, i, '—' if v == 0 else f'{v:.2f}', ha='center', va='center',
                    fontsize=7.5, color='white' if v > 0.6 else MID)

    ax.axvline(3.5, color=INK, lw=1.6)
    ax.set_xticks(range(8)); ax.set_xticklabels(feats, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'{f}  (n={p})' for f, p in zip(files, people)], fontsize=8)
    ax.text(1.5, -0.95, 'actigraphy', ha='center', fontsize=8.5, style='italic')
    ax.text(5.5, -0.95, 'clinical rating scales', ha='center', fontsize=8.5, style='italic')
    ax.grid(False)
    ax.set_title('No participant has both halves of the feature vector',
                 fontsize=10, loc='left', pad=22)
    save(fig, out_dir, 'fig2_obf_coverage')


# ---------------------------------------------------------------- Figure 3
def figure3(R, out_dir):
    """The gap between fusion and its within-class shuffle never opens."""
    W = R['B6_waterfall']
    stages = [k for k in W if k.startswith(('S1', 'S2', 'S3 '))]
    short = ['Repeated CV\n(same features)', '+ clinical scales\nremoved',
             '+ grouped by\nWESAD subject']
    fus = [W[s]['fusion'] for s in stages]
    wic = [W[s]['fusion_within_class_shuffle'] for s in stages]
    acr = [W[s]['fusion_across_class_shuffle'] for s in stages]
    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    for vals, lab, style in [(fus, 'Fusion (aligned)', dict(color=HILITE, marker='o', lw=1.8)),
                             (wic, 'Within-class permutation', dict(color=INK, marker='s',
                                                                   lw=1.4, ls='--')),
                             (acr, 'Across-class permutation', dict(color=MID, marker='^',
                                                                   lw=1.2, ls=':'))]:
        m = [v[0] for v in vals]; s = [v[1] for v in vals]
        ax.errorbar(x, m, yerr=s, capsize=3, **style, label=lab)

    ax.axhline(R['B6_waterfall']['S0 reported protocol (single split, build 42)']['fusion'],
               color=PALE, lw=1.2, ls='-')
    ax.text(2.42, 0.829, 'as reported (0.826)', ha='right', fontsize=7.5, color=MID)
    ax.axhline(1 / 3, ls=':', lw=1, color=MID)
    ax.text(2.42, 0.345, 'chance', ha='right', fontsize=7.5, color=MID)

    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8.5)
    ax.set_xlim(-0.35, 2.55); ax.set_ylim(0.25, 0.95)
    ax.set_ylabel('Macro-F1')
    # opaque box: the chance line would otherwise read straight through the legend text
    ax.legend(loc='lower left', fontsize=8, frameon=True, facecolor='white',
              framealpha=1.0, edgecolor='none', borderpad=0.6)
    ax.set_title('Corrections lower performance; the shuffle gap stays closed',
                 fontsize=10, loc='left', pad=10)
    save(fig, out_dir, 'fig3_correction_waterfall')


# ---------------------------------------------------------------- Figure 4
def figure4(R, out_dir):
    """The reported calibration figure is a bin-count choice."""
    P = R['B7_calibration']['original_protocol']
    bins = [5, 10, 15]
    cal = [P[f'isotonic_fit_on_train_bins{b}'] for b in bins]
    held = R['B7_calibration']['heldout_cv_calibrated_ece10']

    fig, ax = plt.subplots(figsize=(4.6, 3.2))

    # Held-out reference band first, so it sits behind the series.
    ax.fill_between([4, 16], held['p2.5'], held['p97.5'], color=INK, alpha=0.08, lw=0)
    ax.axhline(held['mean'], color=INK, ls='--', lw=1.3)
    # sits directly on the line it describes, left of where the series crosses it
    ax.text(4.25, held['mean'] + 0.006,
            f"held-out estimate: {held['mean']:.3f}",
            ha='left', va='bottom', fontsize=8, color=INK)

    m = [c['ece'] for c in cal]
    lo = [c['ece'] - c['ci95'][0] for c in cal]
    hi = [c['ci95'][1] - c['ece'] for c in cal]
    ax.errorbar(bins, m, yerr=[lo, hi], capsize=4, marker='o', color=HILITE, lw=1.8,
                label='Isotonic, fitted on training rows')

    for b, v in zip(bins, m):
        dx, ha = (-0.35, 'right') if b == bins[-1] else (0.35, 'left')
        ax.text(b + dx, v, f'{v:.3f}', fontsize=8.5, va='center', ha=ha, color=INK)

    ax.text(5, 0.008, 'as reported', ha='center', va='bottom', fontsize=8, style='italic')

    ax.set_xticks(bins); ax.set_xlim(4, 16); ax.set_ylim(0, 0.33)
    ax.set_xlabel('Number of bins'); ax.set_ylabel('Expected calibration error')
    ax.set_title(f"A reporting parameter, not a property of the model",
                 fontsize=10, loc='left', pad=10)
    ax.text(0.015, 0.97, f"ECE on {R['B7_calibration']['n_test']} test rows; "
                         "bars are 95% bootstrap intervals",
            transform=ax.transAxes, va='top', fontsize=7.5, color=MID)

    save(fig, out_dir, 'fig4_calibration_sensitivity')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--as-run', default='run_as_run',
                    help='directory holding the 11-subject results.json')
    ap.add_argument('--complete', default='run_complete',
                    help='directory holding the 15-subject results.json')
    ap.add_argument('--out', default='../figures')
    a = ap.parse_args()

    with open(os.path.join(a.as_run, 'results.json')) as f:
        R = json.load(f)
    print(f'figures from {a.as_run} -> {a.out}')
    figure1(R, a.out)
    figure2(a.out)
    figure3(R, a.out)
    figure4(R, a.out)
    print('done. Figures 1, 3 and 4 use the as-run pass (they describe the original pipeline);')
    print('Figure 2 is structural and identical in both.')


if __name__ == '__main__':
    main()
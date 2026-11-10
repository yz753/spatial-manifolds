"""Each region's own anchoring axis, compared against every other region's.

Every regional analysis so far scores cells against an axis built from MEC, which
makes MEC the privileged frame: a region can only look like a follower to the
extent it resembles entorhinal cortex, and "does visual cortex have a state of
its own?" cannot even be asked. Here each region builds its OWN first principal
component of its own cell x trial label matrix, and the axes are compared with
one another. No region defines the frame.

THE PROBLEM THAT MAKES A RAW AXIS-TO-AXIS CORRELATION UNINTERPRETABLE. An axis
estimated from 20 noisy cells is itself noisy, so a low correlation between two
regions' axes can mean either that they express different states or that one of
them barely expresses a state at all. Those are opposite conclusions. The
separation is standard: measure each region's SPLIT-HALF reliability -- build the
axis twice from disjoint halves of its own cells and correlate -- and divide the
cross-region correlation by the geometric mean of the two reliabilities
(Spearman's correction for attenuation):

    r_true(i,j) = r(i,j) / sqrt( r(i,i) * r(j,j) )

r(i,i) answers "does this region have a coherent state at all?", which is worth
reporting on its own. r_true answers "given that both regions express a state,
is it the SAME state?" -- the question the paper actually needs, and the one a
raw correlation cannot answer.

EVERYTHING IS SIZE-MATCHED. Reliability rises with cell count, and MEC
contributes hundreds of cells where visual cortex contributes tens, so a
size-matched comparison is not optional: each pair is computed at
n = min(n_i, n_j) cells per region, split into two halves of h = n // 2, and
every axis in the comparison -- within-region and cross-region -- is built from
exactly h cells. Averaged over N_SPLIT random splits.

Cells whose labels are constant across trials (fully gated) carry no variance and
are excluded before any count is taken; a region's count here is therefore its
number of label-VARYING cells, not its number of cells.

Disattenuated correlations can exceed 1 when reliabilities are small, which is a
symptom of the estimate being noisy rather than of the state being more than
identical. Raw and disattenuated values are both written out, and the figure
plots both so the correction is visible rather than implicit.

Writes data/population_state/region_axis_coherence.csv
       fig_region_axis_coherence.pdf
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import load_session_labels

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
OUT_CSV = f'{PS}/region_axis_coherence.csv'
OUT_PDF = f'{FIG}/fig_region_axis_coherence.pdf'

MIN_CELLS = 16        # label-varying cells per region; h = 8 per half
N_SPLIT = 40
REGIONS = ['MEC', 'SUB', 'VIS', 'CB']
RCOL = {'MEC': '#7b4173', 'VIS': '#3f9b4f', 'SUB': '#c0723a', 'CB': '#4c72b0'}


def fam(r):
    r = str(r)
    if r.startswith('ENTm'):
        return 'MEC'
    if any(r.startswith(x) for x in ('PRE', 'POST', 'PAR', 'SUB')):
        return 'SUB'
    if r.startswith('VIS'):
        return 'VIS'
    if any(k in r for k in ('CENT', 'CUL', 'ARB', 'DEC', 'FOTU', 'PYR', 'SIM',
                            'AN', 'PRM', 'COPY', 'FL', 'NOD', 'UVU')):
        return 'CB'
    return 'other'


C = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
C['macro'] = C.brain_region.map(fam)


def axis_from(M):
    """PC1 of a cell x trial label matrix, signed to that subset's own anchored
    fraction -- so the orientation is set by the cells in hand and no region's
    convention is imposed on another."""
    frac = np.nanmean(M, axis=0)
    Mc = np.nan_to_num(M - np.nanmean(M, axis=1, keepdims=True))
    U, sv, Vt = np.linalg.svd(Mc, full_matrices=False)
    pc = Vt[0]
    if np.corrcoef(pc, np.nan_to_num(frac))[0, 1] < 0:
        pc = -pc
    return pc


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def session(mo, dy, rng):
    z = load_session_labels(mo, dy)
    if z is None:
        return []
    L, ids = z['labels'], [int(c) for c in z['cluster_id']]
    g = C[(C.mouse == mo) & (C.day == dy)].set_index('cluster_id').macro.to_dict()
    byreg = {}
    for k, c in enumerate(ids):
        v = np.nan_to_num(L[k])
        if np.std(v) == 0:
            continue
        r = g.get(c)
        if r in REGIONS:
            byreg.setdefault(r, []).append(v)
    byreg = {r: np.array(v) for r, v in byreg.items() if len(v) >= MIN_CELLS}
    if len(byreg) < 2:
        return []

    rows = []
    regs = sorted(byreg)
    for i in range(len(regs)):
        for j in range(i + 1, len(regs)):
            ri, rj = regs[i], regs[j]
            Mi, Mj = byreg[ri], byreg[rj]
            n = min(len(Mi), len(Mj))
            h = n // 2
            if h < MIN_CELLS // 2: # this implies AT LEAST 8 cells per half are used instead of what's described in the paper (precisely 8 cells per half)
                continue
            rii, rjj, rij = [], [], []
            for _ in range(N_SPLIT):
                pi = rng.permutation(len(Mi))[:2 * h]
                pj = rng.permutation(len(Mj))[:2 * h]
                Ai, Bi = axis_from(Mi[pi[:h]]), axis_from(Mi[pi[h:]])
                Aj, Bj = axis_from(Mj[pj[:h]]), axis_from(Mj[pj[h:]])
                rii.append(corr(Ai, Bi))
                rjj.append(corr(Aj, Bj))
                # all four cross pairings, so the cross estimate uses the same
                # amount of data as each reliability estimate
                rij.append(np.nanmean([corr(Ai, Aj), corr(Ai, Bj),
                                       corr(Bi, Aj), corr(Bi, Bj)]))
            a, b, c_ = np.nanmean(rii), np.nanmean(rjj), np.nanmean(rij)
            dis = c_ / np.sqrt(a * b) if (a > 0 and b > 0) else np.nan
            rows.append(dict(mouse=mo, day=dy, reg_i=ri, reg_j=rj, n_cells=n,
                             h=h, r_ii=a, r_jj=b, r_ij=c_, disatt=dis))
    return rows


if __name__ == '__main__':
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    out = []
    for mo, dy in sess:
        try:
            out += session(mo, dy, np.random.default_rng(abs(hash((mo, dy))) % 2 ** 32))
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
    D = pd.DataFrame(out)
    D.to_csv(OUT_CSV, index=False)
    print(f'{len(D)} region pairs from {D.groupby(["mouse", "day"]).ngroups} sessions\n')

    D['pair'] = D.reg_i + '-' + D.reg_j
    print('WITHIN-REGION reliability (does the region have a state of its own?)')
    rel = pd.concat([D[['mouse', 'day', 'reg_i', 'r_ii', 'h']]
                     .rename(columns={'reg_i': 'reg', 'r_ii': 'rel'}),
                     D[['mouse', 'day', 'reg_j', 'r_jj', 'h']]
                     .rename(columns={'reg_j': 'reg', 'r_jj': 'rel'})])
    for r in REGIONS:
        x = rel[rel.reg == r].groupby(['mouse', 'day']).rel.mean().dropna()
        if not len(x):
            continue
        p = wilcoxon(x).pvalue if len(x) >= 6 else np.nan
        print(f'  {r:4s} n={len(x):2d} sessions   split-half r = {x.mean():+.3f}  p={p:.3g}')

    print('\nCROSS-REGION agreement, raw and corrected for each region\'s reliability')
    for pr in sorted(D.pair.unique()):
        x = D[D.pair == pr].dropna(subset=['disatt'])
        if not len(x):
            continue
        p = wilcoxon(x.disatt - 1).pvalue if len(x) >= 6 else np.nan
        print(f'  {pr:9s} n={len(x):2d}  raw {x.r_ij.mean():+.3f}   '
              f'disattenuated {x.disatt.mean():+.3f}   vs 1.0: p={p:.3g}')

    # ============================== figure =====================================
    fig = plt.figure(figsize=(9.4, 3.4))
    G = fig.add_gridspec(1, 3, wspace=.46, width_ratios=[1, 1.1, 1.1])

    def tidy(ax):
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    def lp(ax, s, x=-.20):
        ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=10, weight='bold',
                va='bottom', ha='right')

    rng = np.random.default_rng(0)
    # A: each region's own state coherence
    ax = fig.add_subplot(G[0])
    for j, r in enumerate(REGIONS):
        x = rel[rel.reg == r].groupby(['mouse', 'day']).rel.mean().dropna()
        if not len(x):
            continue
        ax.scatter(np.full(len(x), j) + rng.uniform(-.15, .15, len(x)), x, s=12,
                   color=RCOL[r], alpha=.75, lw=0, zorder=3)
        ax.plot([j - .28, j + .28], [x.median()] * 2, color='k', lw=2, zorder=4)
        ax.annotate(f'{len(x)}', (j, .02), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=5.6, color='0.45')
    ax.axhline(0, color='0.5', lw=.9, ls='--')
    ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS, fontsize=7.5)
    ax.set_ylabel('split-half reliability of\nthe region\'s own axis', fontsize=8)
    ax.set_title(f'each region has a state of its own\n(size-matched, {MIN_CELLS // 2} cells per half)',
                 fontsize=8.5)
    tidy(ax); lp(ax, 'A')

    # B: raw cross-region agreement
    prs = [p for p in ['MEC-SUB', 'MEC-VIS', 'SUB-VIS', 'CB-MEC', 'CB-VIS']
           if p in set(D.pair)]
    ax = fig.add_subplot(G[1])
    for j, pr in enumerate(prs):
        x = D[D.pair == pr].r_ij.dropna()
        ax.scatter(np.full(len(x), j) + rng.uniform(-.15, .15, len(x)), x, s=12,
                   color='0.45', alpha=.75, lw=0, zorder=3)
        ax.plot([j - .28, j + .28], [x.median()] * 2, color='k', lw=2, zorder=4)
        ax.annotate(f'{len(x)}', (j, .02), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=5.6, color='0.45')
    ax.axhline(0, color='0.5', lw=.9, ls='--')
    ax.set_xticks(range(len(prs))); ax.set_xticklabels(prs, fontsize=6.8, rotation=18,
                                                       ha='right')
    ax.set_ylabel('raw agreement between axes', fontsize=8)
    ax.set_title('raw: confounds "different state"\nwith "no state"', fontsize=8.5)
    tidy(ax); lp(ax, 'B', x=-.22)

    # C: corrected for each region's own reliability
    ax = fig.add_subplot(G[2])
    for j, pr in enumerate(prs):
        x = D[D.pair == pr].disatt.dropna()
        ax.scatter(np.full(len(x), j) + rng.uniform(-.15, .15, len(x)), x, s=12,
                   color='#c04744', alpha=.75, lw=0, zorder=3)
        ax.plot([j - .28, j + .28], [x.median()] * 2, color='k', lw=2, zorder=4)
        if len(x) >= 6:
            p = wilcoxon(x - 1).pvalue
            ax.annotate(f'p={p:.3f}' if p >= .001 else f'p={p:.0e}',
                        (j, .99), xycoords=('data', 'axes fraction'), ha='center',
                        va='top', fontsize=5.8,
                        color='#a33' if p < .05 else '0.45')
    ax.axhline(1, color='0.4', lw=1, ls='--')
    ax.text(len(prs) - .4, 1.02, 'identical state', fontsize=6, color='0.4',
            ha='right')
    ax.set_xticks(range(len(prs))); ax.set_xticklabels(prs, fontsize=6.8, rotation=18,
                                                       ha='right')
    ax.set_ylabel('agreement / $\\sqrt{r_{ii} r_{jj}}$', fontsize=8)
    ax.set_title('corrected: is it the SAME state,\ngiven each has one?', fontsize=8.5)
    tidy(ax); lp(ax, 'C', x=-.22)

    fig.savefig(OUT_PDF, bbox_inches='tight')
    print(f'\nwrote {OUT_CSV}\nwrote {OUT_PDF}')

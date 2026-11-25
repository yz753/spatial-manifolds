"""Figure 1 supplement: does anchoring outside MEC predict behaviour too?

The paper's population anchoring state is built from entorhinal cells. If the
same state computed from cells in VISUAL CORTEX or the SUBICULAR COMPLEX tracks
behaviour just as well, then "the MEC population state predicts behaviour" is
not a statement about MEC. This is the control for that, and it matters because
three independent analyses already point the same way: cells outside MEC follow
the MEC anchoring axis at least as strongly as cells within it, the Anchoring
Dependence Index is unchanged under an MEC-only population, and the LFP
speed-theta decoupling is not entorhinal-specific either.

The same per-cell labels are used throughout -- the cached build already
classifies every curated cell regardless of region -- so the only thing that
changes between regions is which cells are averaged into the population state.
Nothing is reclassified, so a difference between regions cannot come from a
difference in method.

    A   one session: the population state computed from MEC cells and from
        visual cells, on the same trial axis
    B   how well the two agree, across sessions
    C-E hit rate by trial type and state, with the state defined by MEC,
        visual and subicular cells in turn
    F   discrimination: AUC of the population state predicting a hit, for cued
        and uncued trials, by region
    G   the Anchoring Dependence Index, AUC(uncued) - AUC(cued), by region

Cerebellum is included as the distant negative control -- cells there have no
business tracking an entorhinal anchoring state -- but only 5 sessions carry any
curated cerebellar cells at all and the largest has 18, so its panels are shown
with their n and should not be read as a test.

THIS SUPPLEMENT IS ABOUT BEHAVIOUR PREDICTION, NOT ABOUT FOLLOWING. It asks
whether a population state built from another region predicts the animal's
performance as well as the entorhinal one does, which needs no null: hit rate
and AUC are scored against the behaviour, not against a chance correlation. The
separate question -- whether cells in each region FOLLOW the entorhinal axis
above their own chance level -- is answered in Figure 4B, and it has to be
scored against each cell's own circular-shift null, because chance differs
between regions (+0.130 to +0.157) and raw correlations therefore rank regions
partly on how often their cells happen to be anchored.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pynapple as nap
from scipy.stats import wilcoxon, pearsonr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR, load_session_labels

import json as _json
_g = {}
for _c in _json.load(open('/Users/harryclark/Documents/spatial-manifolds/scripts/'
                          'figures/AnchorDynamics2026/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
clip_trials = _g['clip_trials']

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
PS = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
SRC = '/Volumes/datasetsSD/Ephys_data/clark2025/'
MIN_CELLS, MIN_TRIALS = 15, 30
# Session gating copied from Figure 1 (cell 14 of figure1_schematics.ipynb) so
# the numbers are directly comparable: a session is included when it has at
# least MIN_N trials in EACH of the four trial-type x state cells, and an AUC is
# computed for a trial type when it has at least AUC_MIN trials with a hit rate
# strictly between 0 and 1. The four-cell rule is what makes a session able to
# speak to the question -- it must contain both states within both trial types.
MIN_N, AUC_MIN = 10, 25
# Session gating, matched to Figure 1 so the numbers are directly comparable:
# at least MIN_PER_CELL trials in EACH of the four trial-type x state cells, and
# at least MIN_PER_TYPE trials of each trial type. The four-cell rule is what
# makes a session able to speak to the question at all -- it must contain both
# states within both trial types -- and it is why Figure 1 reports 23 sessions
# where a weaker rule admits 59.
MIN_PER_CELL, MIN_PER_TYPE = 10, 20
MIN_CELLS_CB = 10      # cerebellum never reaches 15; see the note above
REGIONS = ['MEC', 'VIS', 'SUB', 'CB']
RCOL = {'MEC': '#7b4173', 'VIS': '#3f9b4f', 'SUB': '#c0723a', 'CB': '#4c72b0'}
CEREB = ('SIM', 'CUL', 'CENT', 'AN', 'PRM', 'COPY', 'PFL', 'FL', 'DEC', 'FOTU',
         'PYR', 'UVU', 'NOD', 'CB')


def fam(r):
    r = str(r)
    if r.startswith('ENTm'): return 'MEC'
    if r.startswith('VIS'): return 'VIS'
    if r.startswith(('PRE', 'POST', 'PAR', 'SUB')): return 'SUB'
    if r.startswith(CEREB): return 'CB'
    return None


# Regions come from the classification datasheet, NOT from pc1_by_region.csv.
# That file drops cells whose labels have zero variance, which under the gated
# classifier is about a third of all cells -- precisely the fully anchored and
# fully non-anchored ones. Building the region map from it therefore silently
# excluded them and gave an MEC state correlating only 0.63 with the paper's.
# ALL curated cells (GC, NGS, NS), matching the population definition used for
# the paper's state -- not spatial cells only.
REG = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/'
                  'cell_classifications_v2.csv')
REG = REG[REG.cell_class_of1.isin(['GC', 'NGS', 'NS'])]
REG['fam'] = [fam(x) for x in REG.brain_region]
RMAP = {(int(m), int(d), int(c)): f for m, d, c, f in
        REG[['mouse', 'day', 'cluster_id', 'fam']].values}


def session_table(mo, dy):
    """Per-trial behaviour joined to a population state for each region."""
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    stem = f'{SRC}M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    if not os.path.exists(stem):
        return None
    cp = stem.replace('_beh.nwb', '_srt-kilosort4_clusters.npz')
    beh = nap.load_file(stem)
    clusters = nap.load_file(cp)
    # clip_trials RENUMBERS trials from 1 and keeps the original in
    # `orig_number`. The label files store the renumbered ids, so the behaviour
    # must come through the same call -- indexing beh['trials'] by those ids
    # matches them against the ORIGINAL numbering and silently assigns each
    # trial another trial's type and outcome (trial-type agreement 0.52).
    tr, _orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    tr['number'] = tr.number.astype(int)
    # per-trial mean RUNNING speed (>= 3 cm/s), needed for the speed-matching
    # control: hit rate depends on how fast the animal is going, so an
    # anchoring-behaviour link could be a speed effect wearing an anchoring label
    S = beh['S']
    st_, sv_ = np.asarray(S.index), np.asarray(S.values, float)
    keep = tr[tr.number.isin(set(z['trial'].tolist()))].copy()
    if len(keep) < MIN_TRIALS:
        return None
    assert set(keep.number) == set(int(t) for t in z['trial']), 'trial id mismatch'
    spd = {}
    for _, t in tr.iterrows():
        m = (st_ >= t.start) & (st_ <= t.end) & (sv_ >= 3.0)
        spd[int(t.number)] = float(sv_[m].mean()) if m.sum() > 10 else np.nan
    out = pd.DataFrame(dict(mouse=mo, day=dy, trial=z['trial'],
                            speed=[spd.get(int(t), np.nan) for t in z['trial']],
                            ttype=keep.set_index('number').loc[z['trial'], 'type'].values,
                            hit=(keep.set_index('number')
                                 .loc[z['trial'], 'performance'] == 'hit').astype(int).values))
    # MEC state comes from the canonical table, the same column Figure 1 reads,
    # so the two cannot drift apart
    TT = pd.read_csv(f'{PS}/trial_table.csv')
    tt = TT[(TT.mouse == mo) & (TT.day == dy)].set_index('trial')
    out['frac_MEC_canon'] = [tt.frac_anch.get(int(t), np.nan) for t in z['trial']]
    fams = np.array([RMAP.get((mo, dy, int(c))) for c in z['cluster_id']])
    for r in REGIONS:
        m = fams == r
        L = z['labels'][m]
        ok = ~np.isnan(L).all(axis=1)
        need = MIN_CELLS_CB if r == 'CB' else MIN_CELLS
        out[f'frac_{r}'] = np.nanmean(L[ok], axis=0) if ok.sum() >= need else np.nan
        if r == 'MEC':                       # use the canonical column verbatim
            out['frac_MEC'] = out.frac_MEC_canon
        out[f'n_{r}'] = int(ok.sum())
    return out.drop(columns=['frac_MEC_canon'])


sess = sorted({(int(f.split('M')[1].split('D')[0]), int(f.split('D')[1].split('.')[0]))
               for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
tabs = []
for mo, dy in sess:
    try:
        t = session_table(mo, dy)
    except Exception:
        continue
    if t is not None:
        tabs.append(t)
D = pd.concat(tabs, ignore_index=True)
D.to_csv(f'{PS}/nonmec_population_state.csv', index=False)
print(f'{D.groupby(["mouse","day"]).ngroups} sessions, {len(D)} trials')
for r in REGIONS:
    n = D[D[f'frac_{r}'].notna()].groupby(['mouse', 'day']).ngroups
    print(f'  {r}: {n} sessions with >= {MIN_CELLS} cells')

# ── per-session behavioural statistics, one row per region ───────────────────
rows = []
for (mo, dy), d in D.groupby(['mouse', 'day']):
    for r in REGIONS:
        f = d[f'frac_{r}']
        if f.isna().all():
            continue
        rec = dict(mouse=mo, day=dy, region=r, n_cells=int(d[f'n_{r}'].iloc[0]))
        ok = True
        anch = f > .5
        for tt, nm in (('b', 'cued'), ('nb', 'uncued')):
            for st, sn in ((True, 'a'), (False, 'n')):
                m = (d.ttype == tt) & (anch == st) & f.notna()
                if m.sum() < MIN_N:
                    ok = False
                rec[f'hit_{nm}_{sn}'] = d.hit[m].mean() if m.sum() >= MIN_N else np.nan
        for tt, nm in (('b', 'cued'), ('nb', 'uncued')):
            m = (d.ttype == tt) & f.notna()
            y, x = d.hit[m].values, f[m].values
            v = np.isfinite(x) & np.isfinite(y)
            rec[f'auc_{nm}'] = (roc_auc_score(y[v], x[v])
                                if v.sum() >= AUC_MIN and 0 < y[v].mean() < 1 else np.nan)
        if ok:
            rec['adi'] = rec['auc_uncued'] - rec['auc_cued']
            rows.append(rec)
B = pd.DataFrame(rows)
print()
for r in REGIONS:
    b = B[B.region == r]
    if len(b) < 3:
        print(f'  {r}: only {len(b)} sessions, skipped'); continue
    print(f'  {r}: n={len(b)}  AUC cued {b.auc_cued.mean():.3f}  '
          f'uncued {b.auc_uncued.mean():.3f}  ADI {b.adi.mean():+.3f} '
          f'(p={wilcoxon(b.adi).pvalue:.3g})')

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.6, 10.8))
gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.15, 1.15, 1.15],
                      hspace=.62, wspace=.52)

# A: one session, MEC vs VIS state. Chosen for having the most visual cells.
cand = D[D.frac_VIS.notna()].groupby(['mouse', 'day']).n_VIS.first()
MO, DY = map(int, cand.idxmax())
d = D[(D.mouse == MO) & (D.day == DY)]
ax = fig.add_subplot(gs[0, :])
for r in ('MEC', 'VIS'):
    ax.plot(np.arange(1, len(d) + 1), d[f'frac_{r}'].values, color=RCOL[r], lw=1.2,
            label=f'{r} ({int(d[f"n_{r}"].iloc[0])} cells)')
ax.axhline(.5, color='0.6', lw=.7, ls=':')
rv = pearsonr(d.frac_MEC, d.frac_VIS)
ax.set_title(f'M{MO} D{DY} — population state from MEC vs visual cells, '
             f'r = {rv[0]:.2f}', fontsize=9, loc='left')
ax.set_xlabel('Trial', fontsize=8.5); ax.set_ylabel('Fraction anchored', fontsize=8.5)
ax.legend(fontsize=7, frameon=False, ncol=2)
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

# B-D: hit rate by trial type and state, one panel per region
for j, r in enumerate(REGIONS):
    ax = fig.add_subplot(gs[1, j])
    b = B[B.region == r].dropna(subset=['hit_cued_a', 'hit_cued_n',
                                        'hit_uncued_a', 'hit_uncued_n'])
    if len(b) < 3:
        ax.text(.5, .5, f'{r}\ntoo few sessions', ha='center', va='center',
                fontsize=8, color='0.4', transform=ax.transAxes); ax.axis('off')
        continue
    XS, rng = [0, .8, 2.1, 2.9], np.random.default_rng(0)
    vals = [b.hit_cued_a, b.hit_cued_n, b.hit_uncued_a, b.hit_uncued_n]
    cols = [ANCH_COLOR, NONANCH_COLOR, ANCH_COLOR, NONANCH_COLOR]
    for x, v, c in zip(XS, vals, cols):
        ax.bar(x, 100 * v.mean(), width=.78, color=c, lw=.7, edgecolor='k', zorder=1)
        ax.errorbar(x, 100 * v.mean(), yerr=100 * v.std(ddof=1) / np.sqrt(len(v)),
                    color='k', lw=1, capsize=2.5, zorder=4)
        ax.scatter(x + rng.uniform(-.15, .15, len(v)), 100 * v, s=5, color='0.25',
                   alpha=.5, lw=0, zorder=3)
    for x0, x1, va, vn in ((0, .8, b.hit_cued_a, b.hit_cued_n),
                           (2.1, 2.9, b.hit_uncued_a, b.hit_uncued_n)):
        p = wilcoxon(va, vn).pvalue
        star = 'n.s.' if p > .05 else ('*' if p > .01 else ('**' if p > .001 else '***'))
        ax.plot([x0, x0, x1, x1], [104, 107, 107, 104], color='0.3', lw=.8)
        ax.text((x0 + x1) / 2, 108, star, ha='center', fontsize=7, color='0.2')
    ax.set_xticks([.4, 2.5]); ax.set_xticklabels(['cued', 'uncued'], fontsize=7.5)
    ax.set_ylim(0, 120); ax.set_yticks([0, 50, 100])
    ax.set_title(f'state from {r} (n={len(b)})', fontsize=8.5, loc='left',
                 color=RCOL[r])
    if j == 0:
        ax.set_ylabel('Hit rate (%)', fontsize=8.5)
    ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

# E: agreement between MEC and each other region's state
ax = fig.add_subplot(gs[2, 0])
for j, r in enumerate(['VIS', 'SUB', 'CB']):
    rs = []
    for (mo, dy), d2 in D.groupby(['mouse', 'day']):
        if d2[f'frac_{r}'].isna().all() or d2.frac_MEC.isna().all():
            continue
        rs.append(pearsonr(d2.frac_MEC, d2[f'frac_{r}'])[0])
    if not rs:
        continue
    ax.scatter(np.full(len(rs), j) + np.random.default_rng(0).uniform(-.12, .12, len(rs)),
               rs, s=14, color=RCOL[r], lw=0)
    ax.plot([j - .2, j + .2], [np.median(rs)] * 2, color='k', lw=1.8)
    ax.text(j, 1.06, f'n={len(rs)}', ha='center', fontsize=7, color='0.35')
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['VIS', 'SUB', 'CB'], fontsize=8)
ax.set_xlim(-.5, 2.5); ax.set_ylim(-.6, 1.15)
ax.set_ylabel('r with MEC state', fontsize=8.5)
ax.set_title('state agreement', fontsize=8.5, loc='left')
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

# F: AUC by trial type and region
ax = fig.add_subplot(gs[2, 2])
for j, r in enumerate(REGIONS):
    b = B[B.region == r]
    if len(b) < 3:
        continue
    for k, (col, mk) in enumerate((('auc_cued', 'o'), ('auc_uncued', 's'))):
        x = j + (k - .5) * .32
        ax.errorbar(x, b[col].mean(), yerr=b[col].std(ddof=1) / np.sqrt(len(b)),
                    color=RCOL[r], marker=mk, ms=5, lw=1.2, capsize=3,
                    mfc='white' if k == 0 else RCOL[r])
ax.axhline(.5, color='0.6', lw=.7, ls=':')
ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS, fontsize=8)
ax.set_ylabel('AUC: state predicts hit', fontsize=8.5)
ax.set_title('open = cued, filled = uncued', fontsize=8, loc='left')
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

# G: ADI by region
ax = fig.add_subplot(gs[2, 3])
for j, r in enumerate(REGIONS):
    b = B[B.region == r]
    if len(b) < 3:
        continue
    rng = np.random.default_rng(0)
    ax.scatter(np.full(len(b), j) + rng.uniform(-.13, .13, len(b)), b.adi, s=14,
               color=RCOL[r], lw=0, alpha=.75)
    ax.errorbar(j, b.adi.mean(), yerr=b.adi.std(ddof=1) / np.sqrt(len(b)), color='k',
                marker='_', ms=14, lw=1.4, capsize=3)
    p = wilcoxon(b.adi).pvalue
    ax.text(j, ax.get_ylim()[1], 'n.s.' if p > .05 else '*', ha='center', fontsize=7.5)
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS, fontsize=8)
ax.set_ylabel('ADI  (AUC uncued − cued)', fontsize=8.5)
ax.set_title('anchoring dependence', fontsize=8.5, loc='left')
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

h = [plt.Line2D([], [], marker='s', ls='', color=ANCH_COLOR, label='anchored'),
     plt.Line2D([], [], marker='s', ls='', color=NONANCH_COLOR, label='non-anchored')]
fig.legend(handles=h, loc='lower center', ncol=2, fontsize=8, frameon=False,
           bbox_to_anchor=(.5, -.005))


# ── H: the same question asked against each cell's OWN null ──────────────────
# Panels C-G ask whether a population state built from another region predicts
# BEHAVIOUR, which needs no null -- hit rate and AUC are scored against what the
# animal did. This panel asks the different question of whether cells in each
# region FOLLOW the entorhinal axis above their own chance level, and that one
# cannot be read off raw correlations.
#
# A cell anchored on most trials correlates with a population that is also
# anchored on most trials whatever the coordination, and the resulting chance
# level differs BETWEEN REGIONS -- +0.130 in cerebellum to +0.157 for entorhinal
# grid cells. Ranking regions on raw r therefore partly ranks them on how often
# their cells happen to be anchored. Against each cell's own circular-shift null
# the ranking changes: visual cortex, which reaches +0.102 raw and looks like a
# follower, sits BELOW its own null.
NUL = pd.read_csv(f'{PS}/pc1_by_region.csv').merge(
    pd.read_csv(f'{PS}/unit_table.csv')[['mouse', 'day', 'cluster_id', 'identity']],
    on=['mouse', 'day', 'cluster_id'], how='left')
NUL['excess'] = NUL.r - NUL.r_null
NUL['macro'] = NUL.brain_region.map(fam)
NUL = NUL.dropna(subset=['macro', 'excess'])
NUL['sess'] = NUL.mouse.astype(str) + 'D' + NUL.day.astype(str)
NGRP = [('grid\n(MEC)', (NUL.macro == 'MEC') & (NUL.identity == 'grid'), '#c04744'),
        ('int\n(MEC)', (NUL.macro == 'MEC')
         & (NUL.identity == 'putative interneuron'), '#d95f02'),
        ('other\n(MEC)', (NUL.macro == 'MEC')
         & (~NUL.identity.isin(['grid', 'putative interneuron'])), '#888888'),
        ('SUB/\nPARA', NUL.macro == 'SUB', RCOL['SUB']),
        ('VIS', NUL.macro == 'VIS', RCOL['VIS']),
        ('CB', NUL.macro == 'CB', RCOL['CB'])]
ax = fig.add_subplot(gs[3, :3])
_rng = np.random.default_rng(0)
print("\nagreement as excess over each cell's own circular-shift null:")
for j, (nm, m, col) in enumerate(NGRP):
    d = NUL[m]
    per = d.groupby('sess').excess.mean().dropna()
    if not len(per):
        continue
    pw = wilcoxon(per).pvalue if len(per) >= 6 else np.nan
    ax.scatter(np.full(len(per), j) + _rng.uniform(-.15, .15, len(per)), per,
               s=11, color=col, alpha=.75, lw=0, zorder=3)
    ax.plot([j - .27, j + .27], [per.median()] * 2, color='k', lw=2, zorder=4)
    lab = 'not\ntestable' if not np.isfinite(pw) else (
        f'p = {pw:.3f}' if pw >= .001 else f'p = {pw:.0e}')
    ax.annotate(f'{lab}\n{len(per)} sess', (j, .985),
                xycoords=('data', 'axes fraction'), ha='center', va='top',
                fontsize=5.4, linespacing=1.3,
                color='#a33' if (np.isfinite(pw) and pw < .05) else '0.42')
    print(f'  {nm.replace(chr(10), " "):14s} {len(d):5d} cells, {len(per):2d} sess, '
          f'raw {d.r.mean():+.3f}  null {d.r_null.mean():+.3f}  '
          f'excess {d.excess.mean():+.3f}  p={pw:.3g}')
ax.axhline(0, color='0.5', lw=.9, ls='--')
ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
ax.set_xticks(range(len(NGRP)))
ax.set_xticklabels([n for n, _, _ in NGRP], fontsize=6.4)
ax.set_xlim(-.6, len(NGRP) - .4)
ax.set_ylabel('agreement over own null\n(per session)', fontsize=8)
ax.set_title("H  do non-MEC cells FOLLOW the entorhinal axis? Against each cell's own\n"
             "null only the entorhinal groups exceed chance — visual cortex sits BELOW its own",
             fontsize=6.8, loc='left')
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

# the raw-to-null shift drawn directly: that reversal is the point
ax = fig.add_subplot(gs[3, 3])
for nm, m, col in NGRP:
    d = NUL[m]
    ax.plot([0, 1], [d.r.mean(), d.excess.mean()], color=col, lw=1.4,
            marker='o', ms=4, zorder=3)
ax.axhline(0, color='0.5', lw=.9, ls='--')
ax.set_xticks([0, 1]); ax.set_xticklabels(['raw r', 'over\nnull'], fontsize=7)
ax.set_xlim(-.3, 1.3)
ax.set_ylabel('agreement', fontsize=8)
ax.set_title('what the\nnull changes', fontsize=7.5, loc='left')
ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)

out = f'{FIG}/fig1_supp_nonmec_population_state.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')

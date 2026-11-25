"""Open-field decomposition of cells into position (P), head direction (H) and
speed (S) coding, or combinations of them: the Hardcastle et al. (2017)
procedure with XGBoost in place of the LN model.

Usage:  python3 of_xgb_decomp.py worker n_workers
        python3 of_xgb_decomp.py                      (collect, after the workers)
        run_of_xgb.sh does both.

Kept from Hardcastle, Maheswaranathan, Ganguli & Giocomo (2017, Neuron 94,
375-387; code at GiocomoLab/ln-model-of-mec-neurons):
  - spike counts in 20 ms bins, Poisson likelihood
  - 10-fold cross-validation, each fold 5 chunks spread through the session
    (their 50-section scheme), so every fold samples the whole session
  - performance: test log-likelihood increase over a constant-rate model fit
    on the training folds, in bits per spike
  - forward selection: the best single-variable model, then the best model
    adding one variable, kept only if better over the 10 folds (one-sided
    Wilcoxon signed-rank, p < 0.05), and so on; the selected model must beat
    the constant-rate model (signed-rank, p < 0.05) or the cell is unclassified

Replaced: the LN model (one-hot tuning curves under a roughness penalty) by a
Poisson gradient-boosted tree model (xgboost, count:poisson) on the raw
variables. Tree depth is each model's number of input columns (P 2, H 2, S 1,
PH 4, PHS 5). That is the deepest that adds anything: a path of
depth d tests at most min(d, p) of p columns, so a deeper tree only re-splits
columns already used, and a sum of depth-p trees already spans every
piecewise-constant function on a grid of thresholds (a box indicator is a
product of p interval indicators, which expands into 2^p products of one-sided
steps, each a depth-p tree). Depth beyond p therefore changes only how many
trees a fit needs, not what it can fit. The number of trees is set inside each
fold, without the test fold, by early stopping on one held-out training fold
(the work the LN model's roughness penalty does): up to 1,000 trees, stopping
after 30 that fail to improve the validation likelihood, so that shallow
models, which need many trees, are not stopped before they converge.

Variables per 20 ms bin, from OF1 (the session the cell classes come from;
tracking at ~16 Hz, linearly interpolated): P = (x, y); H = (sin, cos) of head
direction, since a tree cannot split across the wrap at +-pi; S = speed.
Hardcastle's fourth variable, theta phase, is not used: the open-field theta
series is not part of the local dataset.

Only the models forward selection visits are fit (at most 6 of the 7).
Cells: curated GC/NGS/NS cells of every labelled session, grouped by the region
families of Figure S3. Writes data/of_xgb/M{m}D{d}.csv and, when run without
arguments, data/of_xgb/of_xgb_cells.csv.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap                                                   # noqa: E402

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
SRC = '/Volumes/datasetsSD/Ephys_data/clark2025'
OUT = f'{ROOT}/data/of_xgb'
os.makedirs(OUT, exist_ok=True)
DT = .02
N_FOLD, N_CHUNK = 10, 5
ALPHA = .05
MIN_SPIKES = 100
XGB = dict(objective='count:poisson', n_estimators=1000, learning_rate=.1,
           min_child_weight=10, subsample=.8, tree_method='hist', n_jobs=1,
           max_delta_step=.7, early_stopping_rounds=30, eval_metric='poisson-nloglik')
CEREB = ('SIM', 'CUL', 'CENT', 'AN', 'PRM', 'COPY', 'PFL', 'FL', 'DEC', 'FOTU',
         'PYR', 'UVU', 'NOD', 'CB')


def fam(r):
    """Region family, as Figure S3 (fig1_supp_nonmec.fam)."""
    r = str(r)
    if r.startswith('ENTm'): return 'MEC'
    if r.startswith('VIS'): return 'VIS'
    if r.startswith(('PRE', 'POST', 'PAR', 'SUB')): return 'SUB'
    if r.startswith(CEREB): return 'CB'
    return None


def session_bins(mo, dy):
    stem = f'{SRC}/M{mo}/D{dy}/OF1/sub-M{mo}_ses-D{dy}_typ-OF1'
    if not os.path.exists(f'{stem}_beh.nwb'):
        return None
    beh = nap.load_file(f'{stem}_beh.nwb')
    cl = nap.load_file(f'{stem}_srt-kilosort4_clusters.npz')
    t = np.asarray(beh['S'].index, float)
    edges = np.arange(max(0., t[0]), t[-1], DT)
    tc = edges[:-1] + DT / 2
    it = lambda k: np.interp(tc, np.asarray(beh[k].index, float), np.asarray(beh[k].values, float))
    h = np.asarray(beh['H'].values, float); th = np.asarray(beh['H'].index, float)
    F = {'P': np.c_[it('P_x'), it('P_y')],
         'H': np.c_[np.interp(tc, th, np.sin(h)), np.interp(tc, th, np.cos(h))],
         'S': it('S')[:, None]}
    sec = np.minimum((np.arange(len(tc)) * N_FOLD * N_CHUNK) // len(tc), N_FOLD * N_CHUNK - 1)
    return dict(edges=edges, F=F, vars=['P', 'H', 'S'], fold=sec % N_FOLD, cl=cl)


def fold_llh(y, X, fold):
    """Test log-likelihood gain over the constant rate, bits/spike, per fold;
    depth = number of input columns, trees by early stopping on a training fold."""
    out = np.full(N_FOLD, np.nan)
    for k in range(N_FOLD):
        tr, te = fold != k, fold == k
        n = y[te].sum()
        if n == 0 or y[tr].sum() == 0:
            continue
        va = fold == (k + 1) % N_FOLD
        fi = tr & ~va
        best = xgb.XGBRegressor(max_depth=X.shape[1], **XGB).fit(
            X[fi], y[fi], eval_set=[(X[va], y[va])], verbose=False)
        r0 = y[tr].mean()
        mu = np.clip(best.predict(X[te]), 1e-9, None)
        ll = (y[te] * np.log(mu) - mu).sum()
        ll0 = (y[te] * np.log(r0) - r0).sum()
        out[k] = (ll - ll0) / n / np.log(2)
    return out


def select(y, F, vars_, fold):
    """Hardcastle forward selection, fitting only the models it visits."""
    LL = {}
    get = lambda m: LL.setdefault(m, fold_llh(y, np.hstack([F[v] for v in m]), fold))
    better = lambda a, b: (np.isfinite(a - b).sum() >= 5 and
                           wilcoxon(a, b, alternative='greater', nan_policy='omit').pvalue < ALPHA)
    cur = max([(v,) for v in vars_], key=lambda m: np.nanmean(get(m)))
    while len(cur) < len(vars_):
        cand = [tuple(v for v in vars_ if v in cur or v == w) for w in vars_ if w not in cur]
        nxt = max(cand, key=lambda m: np.nanmean(get(m)))
        if better(LL[nxt], LL[cur]):
            cur = nxt
        else:
            break
    a = LL[cur]; a = a[np.isfinite(a)]
    ok = len(a) >= 5 and wilcoxon(a, alternative='greater').pvalue < ALPHA
    return (''.join(cur) if ok else ''), LL


def run_session(mo, dy, cells):
    path = f'{OUT}/M{mo}D{dy}.csv'
    if os.path.exists(path):
        return
    B = session_bins(mo, dy)
    if B is None:
        print(f'M{mo}D{dy}: skipped (no OF1)', flush=True)
        return
    rows = []
    for c, region in cells:
        if c not in B['cl'].index:
            continue
        y = np.histogram(np.asarray(B['cl'][c].index, float), B['edges'])[0].astype(float)
        if y.sum() < MIN_SPIKES:
            continue
        model, LL = select(y, B['F'], B['vars'], B['fold'])
        rec = dict(mouse=mo, day=dy, cluster_id=c, region=region, n_spikes=int(y.sum()),
                   rate=y.sum() / (len(y) * DT), model=model, vars=''.join(B['vars']))
        rec.update({f'llh_{"".join(m)}': np.nanmean(v) for m, v in LL.items()})
        rows.append(rec)
    pd.DataFrame(rows).to_csv(path + '.part', index=False)
    os.replace(path + '.part', path)
    print(f'M{mo}D{dy}: {len(rows)} cells', flush=True)


def sessions():
    C = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
    C = C[C.cell_class_of1.isin(['GC', 'NGS', 'NS'])]
    C['fam'] = [fam(x) for x in C.brain_region]
    C = C[C.fam.notna()]
    lab = sorted({(int(f.split('M')[1].split('D')[0]), int(f.split('D')[1].split('.')[0]))
                  for f in os.listdir(f'{ROOT}/data/population_state/labels') if f.endswith('.npz')})
    return [(mo, dy, list(zip(c.cluster_id.astype(int), c.fam)))
            for mo, dy in lab for c in [C[(C.mouse == mo) & (C.day == dy)]] if len(c)]


if __name__ == '__main__':
    S = sessions()
    if len(sys.argv) == 3:
        w, nw = int(sys.argv[1]), int(sys.argv[2])
        # largest sessions first across workers, so they finish together
        S = sorted(S, key=lambda s: -len(s[2]))
        for i, (mo, dy, cells) in enumerate(S):
            if i % nw == w:
                try:
                    run_session(mo, dy, cells)
                except Exception as e:
                    print(f'M{mo}D{dy} failed: {e!r}', flush=True)
    else:
        A = pd.concat([pd.read_csv(f'{OUT}/M{mo}D{dy}.csv') for mo, dy, _ in S
                       if os.path.exists(f'{OUT}/M{mo}D{dy}.csv')], ignore_index=True)
        A['model'] = A.model.fillna('')
        keep = pd.DataFrame([(mo, dy, c) for mo, dy, cells in S for c, _ in cells],
                            columns=['mouse', 'day', 'cluster_id'])
        A = A.merge(keep, on=['mouse', 'day', 'cluster_id'])  # cells excluded since the fit drop out
        A.to_csv(f'{OUT}/of_xgb_cells.csv', index=False)
        print(f'{len(A)} cells, {A.groupby(["mouse", "day"]).ngroups} sessions, '
              f'variables {sorted(A["vars"].unique())}')
        print(pd.crosstab(A.region, A.model, normalize='index').round(3))

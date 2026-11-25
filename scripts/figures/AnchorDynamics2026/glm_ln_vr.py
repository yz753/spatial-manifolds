"""Hardcastle et al. (2017) LN-model decomposition of VR cells into position (P),
speed (S) and theta phase (T) coding, or combinations of them.

Usage:  python3 glm_ln_vr.py [worker n_workers]

Follows the published model (Hardcastle, Maroni & Giocomo 2017, Neuron; code at
GiocomoLab/ln-model-of-mec-neurons) as closely as the virtual track allows:

  - spike counts in 20 ms bins; rate = exp(sum of one tuning parameter per
    variable, read from the bin the animal is in); Poisson likelihood
  - one-hot bins: position 5 cm (their 20 x 20 bins of a 1 m box; here 40 bins
    of the 200 cm track), speed 10 bins, theta phase 18 bins
  - roughness penalty beta/2 * sum of squared differences between neighbouring
    bins, beta 8 for position and 50 for speed and theta (their values),
    circular for theta
  - all 7 models (P, S, T, PS, PT, ST, PST), 10-fold cross-validation, each
    fold 5 chunks spread through the session (their 50-section scheme)
  - performance: test log-likelihood increase over a constant-rate model, in
    bits per spike
  - forward selection: the best single-variable model, then the best model
    adding one variable, kept only if better over folds (one-sided Wilcoxon
    signed-rank, p < 0.05), and so on; the selected model must beat the
    constant-rate model (signed-rank, p < 0.05) or the cell is unclassified

VR changes, and why: speed bins span 0-80 cm/s, not 0-50, because mice run
faster on the track (median 25, 95th percentile 75 cm/s) and a 50 cm/s ceiling
would lump a fifth of the time into the top bin; faster samples are clipped to
the top bin, as theirs were. Head direction is not a VR variable. Time is
restricted to the trials that carry anchoring labels. Theta phase is that of
each cell's own channel group (clark2025_lfp Processed/theta, 1 kHz).

Because every regressor is one-hot, the likelihood depends on the data only
through the spike count and occupancy of each joint (P, S, T) bin, so each
fold is fit on those sums: exact, and fast enough to fit every cell.

Cells: curated GC/NGS/NS cells of every labelled session, grouped by the region
families of Figure S3 (MEC, VIS, SUB, CB). Writes data/glm_ln/M{m}D{d}.csv,
then data/glm_ln/glm_ln_cells.csv when run without arguments after the workers.
"""
import itertools
import os
import sys
import warnings

import h5py
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pynapple as nap                                                   # noqa: E402

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
SRC = '/Volumes/datasetsSD/Ephys_data/clark2025'
LFP = '/Volumes/datasetsSD/Ephys_data/clark2025_lfp'
OUT = f'{ROOT}/data/glm_ln'
os.makedirs(OUT, exist_ok=True)
FS = 1000.0
DT = .02                                       # 20 ms bins
TL = 200.
NP, NS, NT = 40, 10, 18
S_MAX = 80.
BETA = {'P': 8., 'S': 50., 'T': 50.}
NBINS = {'P': NP, 'S': NS, 'T': NT}
N_FOLD, N_CHUNK = 10, 5
ALPHA = .05
VARS = ('P', 'S', 'T')
MODELS = [m for k in (1, 2, 3) for m in itertools.combinations(VARS, k)]
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


def penalty(nb, circular):
    """D'D for the first-difference roughness penalty."""
    D = np.diff(np.eye(nb), axis=0)
    if circular:
        D = np.vstack([D, np.eye(nb)[0] - np.eye(nb)[-1]])
    return D.T @ D


PEN = {'P': penalty(NP, False), 'S': penalty(NS, False), 'T': penalty(NT, True)}


def design(model, idx):
    """One-hot design over the occupied joint bins for the model's variables."""
    blocks, pen = [], []
    for v in model:
        X = np.zeros((len(idx[v]), NBINS[v])); X[np.arange(len(idx[v])), idx[v]] = 1
        blocks.append(X); pen.append(BETA[v] * PEN[v])
    nb = sum(b.shape[1] for b in blocks)
    Pm = np.zeros((nb, nb)); o = 0
    for p in pen:
        Pm[o:o + len(p), o:o + len(p)] = p; o += len(p)
    return np.hstack(blocks), Pm


def fit(X, Pm, y, n, iters=60):
    """Penalised Poisson fit on joint-bin sums: y spikes, n time bins per row."""
    # start from the constant rate, split evenly over the model's variables
    r0 = max(y.sum() / n.sum(), 1e-6)
    w = np.full(X.shape[1], np.log(r0) / X[0].sum())
    ridge = 1e-6 * np.eye(len(w))
    f_old = np.inf
    for _ in range(iters):
        eta = X @ w; mu = n * np.exp(eta)
        f = (mu - y * eta).sum() + .5 * w @ Pm @ w
        g = X.T @ (mu - y) + Pm @ w
        H = (X * mu[:, None]).T @ X + Pm + ridge
        step = np.linalg.solve(H, g)
        t = 1.
        while t > 1e-4:
            w2 = w - t * step; e2 = X @ w2
            f2 = (n * np.exp(e2) - y * e2).sum() + .5 * w2 @ Pm @ w2
            if f2 <= f: break
            t /= 2
        w = w2
        if abs(f_old - f2) < 1e-7 * max(1, abs(f2)): break
        f_old = f2
    return w


def session_bins(mo, dy):
    """Per 20 ms bin within labelled trials: P, S, T-group phase, and spike counts."""
    import fig2_supp_vis_anchored as F
    from spatial_manifolds.anchoring import load_session_labels
    z = load_session_labels(mo, dy)
    bp, cp = F.vr_paths(mo, dy)
    lf = f'{LFP}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    if z is None or not (os.path.exists(bp) and os.path.exists(lf)):
        return None
    beh = nap.load_file(bp); cl = nap.load_file(cp)
    ta = beh['trials'].as_dataframe()
    _, orig = F.clip_trials(ta, cl)
    tk = ta[np.isin(ta.number.values.astype(int), orig)]
    with h5py.File(lf, 'r') as f:
        if 'processing/ecephys/Processed/theta/data' not in f:
            return None
        names = [n.decode() for n in f['general/extracellular_ephys/electrodes/channel_name'][:]]
        nT = f['processing/ecephys/Processed/theta/data'].shape[0]
        ch2g = {c: i for i, n in enumerate(names) for c in n.split('_')}
        tmax = nT / FS
        edges = np.arange(0, tmax, DT)
        tc = edges[:-1] + DT / 2
        intr = np.zeros(len(tc), bool)
        for s, e in zip(tk.start, tk.end):
            intr |= (tc >= s) & (tc < e)
        tc, gidx = tc[intr], np.flatnonzero(intr)
        PH = f['processing/ecephys/Processed/theta/data'][:]
    ph = PH[np.clip(np.round(tc * FS).astype(int), 0, nT - 1)]
    del PH
    P, S = beh['P'], beh['S']
    pi_, si_ = np.asarray(P.index), np.asarray(S.index)
    pos = np.asarray(P.values, float)[np.searchsorted(pi_, tc).clip(0, len(pi_) - 1)]
    spd = np.asarray(S.values, float)[np.searchsorted(si_, tc).clip(0, len(si_) - 1)]
    ok = np.isfinite(pos) & np.isfinite(spd)
    tc, gidx, pos, spd, ph = tc[ok], gidx[ok], pos[ok], spd[ok], ph[ok]
    ip = np.clip((pos / (TL / NP)).astype(int), 0, NP - 1)
    is_ = np.clip((np.clip(spd, 0, None) / (S_MAX / NS)).astype(int), 0, NS - 1)
    # folds: N_FOLD * N_CHUNK contiguous sections, fold k gets every N_FOLD-th
    sec = np.minimum((np.arange(len(tc)) * N_FOLD * N_CHUNK) // len(tc), N_FOLD * N_CHUNK - 1)
    fold = sec % N_FOLD
    # kept bin position of every bin on the regular 20 ms grid (-1: not kept)
    where = np.full(int(tmax / DT) + 1, -1); where[gidx] = np.arange(len(gidx))
    return dict(tc=tc, where=where, ip=ip, is_=is_, ph=ph, fold=fold, cl=cl, ch2g=ch2g)


def cell_models(y, ip, is_, it, fold):
    """Test log-likelihood gain (bits/spike) per model and fold."""
    jk = (ip * NS + is_) * NT + it
    K = NP * NS * NT
    allidx = np.arange(K)
    full = {'P': allidx // (NS * NT), 'S': (allidx // NT) % NS, 'T': allidx % NT}
    LL = {m: np.full(N_FOLD, np.nan) for m in MODELS}
    for k in range(N_FOLD):
        tr, te = fold != k, fold == k
        ytr = np.bincount(jk[tr], y[tr], K); ntr = np.bincount(jk[tr], None, K)
        yte = np.bincount(jk[te], y[te], K); nte = np.bincount(jk[te], None, K)
        nsp = yte.sum()
        if nsp == 0 or ytr.sum() == 0:
            continue
        r0 = ytr.sum() / ntr.sum()
        occ = ntr > 0
        ll0 = (yte * np.log(r0) - nte * r0).sum()
        for m in MODELS:
            idx = {v: full[v][occ] for v in m}
            X, Pm = design(m, idx)
            w = fit(X, Pm, ytr[occ], ntr[occ])
            # test bins: predicted from the weights of their own variable bins
            idt = {v: full[v] for v in m}
            Xt, _ = design(m, idt)
            eta = Xt @ w
            ll = (yte * eta - nte * np.exp(eta)).sum()
            LL[m][k] = (ll - ll0) / nsp / np.log(2)
    return LL


def select(LL):
    """Hardcastle forward selection; returns the model name or '' (unclassified)."""
    mean = {m: np.nanmean(v) for m, v in LL.items()}
    sig = lambda a, b: (np.isfinite(a - b).sum() >= 5
                        and wilcoxon(a, b, alternative='greater', nan_policy='omit').pvalue < ALPHA)
    cur = max([m for m in MODELS if len(m) == 1], key=lambda m: mean[m])
    while len(cur) < 3:
        nxt = max([m for m in MODELS if len(m) == len(cur) + 1 and set(cur) <= set(m)],
                  key=lambda m: mean[m])
        if sig(LL[nxt], LL[cur]):
            cur = nxt
        else:
            break
    a = LL[cur]
    ok = (np.isfinite(a).sum() >= 5
          and wilcoxon(a[np.isfinite(a)], alternative='greater').pvalue < ALPHA)
    return ''.join(cur) if ok else ''


def run_session(mo, dy, cells):
    path = f'{OUT}/M{mo}D{dy}.csv'
    if os.path.exists(path):
        return
    B = session_bins(mo, dy)
    if B is None:
        pd.DataFrame(columns=['mouse']).to_csv(path, index=False)
        return
    cl, where, nb = B['cl'], B['where'], len(B['tc'])
    ec = cl.metadata['extremum_channel']
    rows = []
    for c, region in cells:
        if c not in cl.index or B['ch2g'].get(str(ec[c]), -1) < 0:
            continue
        g = (np.asarray(cl[c].index, float) / DT).astype(int)
        k = where[g[(g >= 0) & (g < len(where))]]
        y = np.bincount(k[k >= 0], None, nb).astype(float)
        if y.sum() < 100:
            continue
        ph = B['ph'][:, B['ch2g'][str(ec[c])]]
        it = np.clip(((np.mod(ph, 2 * np.pi)) / (2 * np.pi / NT)).astype(int), 0, NT - 1)
        LL = cell_models(y, B['ip'], B['is_'], it, B['fold'])
        rec = dict(mouse=mo, day=dy, cluster_id=c, region=region, n_spikes=int(y.sum()),
                   rate=y.sum() / (len(y) * DT), model=select(LL))
        rec.update({f'llh_{"".join(m)}': np.nanmean(v) for m, v in LL.items()})
        rows.append(rec)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'M{mo}D{dy}: {len(rows)} cells', flush=True)


def sessions():
    C = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
    C = C[C.cell_class_of1.isin(['GC', 'NGS', 'NS'])]
    C['fam'] = [fam(x) for x in C.brain_region]
    C = C[C.fam.notna()]
    lab = sorted({(int(f.split('M')[1].split('D')[0]), int(f.split('D')[1].split('.')[0]))
                  for f in os.listdir(f'{ROOT}/data/population_state/labels') if f.endswith('.npz')})
    out = []
    for mo, dy in lab:
        c = C[(C.mouse == mo) & (C.day == dy)]
        if len(c):
            out.append((mo, dy, list(zip(c.cluster_id.astype(int), c.fam))))
    return out


if __name__ == '__main__':
    S = sessions()
    if len(sys.argv) == 3:
        w, nw = int(sys.argv[1]), int(sys.argv[2])
        for i, (mo, dy, cells) in enumerate(S):
            if i % nw == w:
                try:
                    run_session(mo, dy, cells)
                except Exception as e:
                    print(f'M{mo}D{dy} failed: {e!r}', flush=True)
    else:
        A = pd.concat([pd.read_csv(f'{OUT}/M{mo}D{dy}.csv') for mo, dy, _ in S
                       if os.path.exists(f'{OUT}/M{mo}D{dy}.csv')], ignore_index=True)
        A = A.dropna(subset=['cluster_id'])
        A.to_csv(f'{OUT}/glm_ln_cells.csv', index=False)
        print(f'{len(A)} cells, {A.groupby(["mouse", "day"]).ngroups} sessions')
        print(pd.crosstab(A.region, A.model.fillna(''), normalize='index').round(3))

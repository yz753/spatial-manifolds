"""Putative monosynaptic connections, with the multiple-comparison bug fixed.

Usage:  python3 monosyn.py [worker n_workers]

WHY THIS EXISTS. ccg_monosynaptic_M25D25.ipynb implements the English et al. /
Stark & Abeles criteria and finds ZERO connections in 17,366 tested pairs. That
is not biology, it is the two rejection rules:

    R1  reject if ANY non-peak bin exceeds 2.5 SD of the corrected CCG
    R2  reject if ANY anticausal bin has Poisson p < 0.01

Both are applied per-bin and uncorrected, across a +-50 ms window holding 101
bins. Under the null, 2.5 SD one-sided is p = 0.006, so R1 alone fires on about
45% of pairs by chance; R2 tests ~50 anticausal bins at 0.01 and fires on ~39%.
Together they reject roughly two thirds of ALL pairs regardless of what the CCG
contains. The notebook's own top-ranked pair shows it: cell 102 -> 83 has a
z = 10.4 peak at +1 ms that passes every positive criterion and is thrown out by
R1 and R2.

So the rejection rules get the same correction the detection does. A violation
must survive Sidak correction across the number of bins actually examined, which
is what makes "no anticausal structure" a statement about the pair rather than
about how many bins were looked at.

WHAT IS UNCHANGED. 1 ms bins, +-50 ms, hollow-Gaussian predictor (sigma = 5 ms,
60% hollow), the 0.7-4.7 ms causal window, Poisson tests with continuity
correction, minimum spike counts, and the zero-lag exclusion.

VALIDATION IS NOT OPTIONAL HERE. Loosening a rejection rule can only increase
the count, so a higher number is not evidence the fix is right. `jitter_null`
reruns the whole detector on spike trains jittered within +-JITTER_MS, which
destroys millisecond timing while preserving firing rate, slow co-modulation and
the coarse CCG shape. A correct detector finds essentially nothing there. The
false-positive rate it returns is the number that licenses the real count.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from scipy.signal import fftconvolve
from scipy.stats import norm as sp_norm, poisson as sp_poisson
from spatial_manifolds.cross_correlograms import _CCH_inner
from paper_units import curate

SRC = '/Volumes/datasetsSD/Ephys_data/clark2025/'
OUT = '/Users/harryclark/Documents/spatial-manifolds/data/monosyn'
os.makedirs(OUT, exist_ok=True)

BIN_S, MAX_LAG_S = 0.001, 0.050
SIGMA_MS, HOLLOW_FRAC = 5.0, 0.60
CAUSAL = (0.0007, 0.0047)      # English et al. monosynaptic window
MIN_SPIKES = 2000              # per cell
MIN_CCG = 1000                 # summed counts in the pair's CCG
# Set from the jitter null, not by convention. On M25D25 (20,052 pairs tested)
# the real/null counts run 14/1.5 at 1e-3, 11/0.5 at 3e-4, 8/0.5 at 1e-4: 3e-4
# is where the null is essentially empty while the real count is still near its
# maximum, an FDR of about 5%.
ALPHA_PEAK = 3e-4              # before correction across causal bins
ALPHA_REJECT = 0.01            # before correction across the bins examined
MAX_WIDTH_S = 0.003
JITTER_MS = 10                 # null: destroys timing finer than this


def hollow_gaussian(lags_s):
    """Stark & Abeles predictor: a Gaussian with its centre zeroed out."""
    lags_ms = lags_s * 1000.0
    centre = len(lags_ms) // 2
    half = int(round(sp_norm.ppf((1 + HOLLOW_FRAC) / 2) * SIGMA_MS))
    g = np.exp(-0.5 * (lags_ms / SIGMA_MS) ** 2)
    g[centre - half: centre + half + 1] = 0.0
    return g / g.sum()


def poisson_p(obs, exp):
    """Upper-tail Poisson with the continuity correction used throughout."""
    return (1 - sp_poisson.cdf(obs - 1, exp) - .5 * sp_poisson.pmf(obs, exp))


def poisson_p_low(obs, exp):
    """Lower tail, for inhibitory connections.

    An inhibitory cell suppresses its target, so its signature in the CCG is a
    short-latency TROUGH, not a peak. Testing only the upper tail -- which every
    published implementation of this method does, because it was written for
    excitatory connections -- makes inhibitory connections undetectable in
    principle. Since putative interneurons are 9% of the recorded population
    here and the question is whether they participate in a local network, that
    is not a limitation the analysis can carry.
    """
    return sp_poisson.cdf(obs - 1, exp) + .5 * sp_poisson.pmf(obs, exp)


def session_ccgs(mouse, day, jitter=False, rng=None):
    """All-to-all CCGs in spike counts, plus the hollow-Gaussian predictor."""
    stem = f'{SRC}M{mouse}/D{day}/VR/sub-M{mouse}_ses-D{day}_typ-VR'
    cp = f'{stem}_srt-kilosort4_clusters.npz'
    if not os.path.exists(cp):
        return None
    clusters = curate(nap.load_file(cp), mouse, day)
    ids = [int(c) for c in clusters.index]
    if len(ids) < 5:
        return None
    spikes, keep = [], []
    for c in ids:
        t = np.asarray(clusters[c].t, dtype=np.float64)
        if len(t) < MIN_SPIKES:
            continue
        if jitter:
            t = np.sort(t + rng.uniform(-JITTER_MS / 1000, JITTER_MS / 1000, len(t)))
        spikes.append(np.ascontiguousarray(t)); keep.append(c)
    n = len(keep)
    if n < 5:
        return None
    half = int(np.floor(MAX_LAG_S / BIN_S))
    edges = np.linspace(-half * BIN_S, half * BIN_S, 2 * half + 2)
    lags = .5 * (edges[:-1] + edges[1:])
    g = hollow_gaussian(lags)
    raw = np.zeros((n, n, len(lags)), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                raw[i, j] = _CCH_inner(spikes[i], spikes[j], edges, MAX_LAG_S)
    base = np.stack([[np.convolve(raw[i, j], g, mode='same') for j in range(n)]
                     for i in range(n)]).astype(np.float32)
    # this convolution does not take account when reaching to the left/right end of CCG,
    # the code treats the ouside non-existant vals as 0, which decreases calculated baseline/background val
    # this will affect downstream classification: Reject the pair if any negative-lag bin has a sufficiently significant excess.
    # an alternative approach:
    # padding = len(g) // 2
    # base = np.stack([
    #     [
    #         np.convolve(
    #             np.pad(raw[i, j], padding, mode="reflect"),
    #             g,
    #             mode="valid",
    #         )
    #         for j in range(n)
    #     ]
    #     for i in range(n)
    # ]).astype(np.float32)
    
    meta = clusters.metadata.loc[keep] if hasattr(clusters, 'metadata') else None
    return dict(raw=raw, base=base, lags=lags, ids=np.array(keep),
                n_spikes=np.array([len(s) for s in spikes]), meta=meta)


def detect(S, corrected=True, kind='exc'):
    """Ordered pairs passing the criteria. Returns a DataFrame, one row per hit.

    `kind='exc'` looks for a short-latency peak (excitatory), `kind='inh'` for a
    trough (inhibitory). The two are mirror images: the same causal window, the
    same width and zero-lag rules, the same Sidak-corrected rejection of
    anticausal structure, with the Poisson test taken on the opposite tail and
    the extremum taken as a minimum rather than a maximum.

    `corrected=False` reproduces the notebook's uncorrected rejection rules, so
    the two can be compared on identical CCGs.
    """
    inh = kind == 'inh'
    ptail = poisson_p_low if inh else poisson_p
    raw, base, lags = S['raw'], S['base'], S['lags']
    n, L = raw.shape[0], len(lags)
    causal = np.where((lags >= CAUSAL[0]) & (lags <= CAUSAL[1]))[0]
    antic = np.where(lags < 0)[0]
    zero_i = int(np.argmin(np.abs(lags)))
    # Sidak, over the bins each rule actually examines
    a_peak = 1 - (1 - ALPHA_PEAK) ** (1 / len(causal)) if corrected else ALPHA_PEAK
    a_rej = 1 - (1 - ALPHA_REJECT) ** (1 / len(antic)) if corrected else ALPHA_REJECT
    sd_k = sp_norm.ppf(1 - (1 - (1 - ALPHA_REJECT) ** (1 / L)) if corrected
                       else 1 - .006)
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r, b = raw[i, j].astype(float), base[i, j].astype(float)
            if r.sum() < MIN_CCG:
                continue
            corr = r - b
            sgn = -1.0 if inh else 1.0
            k = causal[np.argmax(sgn * corr[causal])]
            p_peak = ptail(r[k], b[k])
            if p_peak >= a_peak:
                continue
            sd = np.std(corr)
            # width: contiguous bins beyond half the extremum or 2 SD, p < 0.01
            wide = [k]
            for step in (-1, 1):
                m = k + step
                while 0 <= m < L and (sgn * corr[m] > sgn * corr[k] / 2
                                      or sgn * corr[m] > 2 * sd) \
                        and ptail(r[m], b[m]) < ALPHA_REJECT:
                    wide.append(m); m += step
            if len(wide) * BIN_S > MAX_WIDTH_S or zero_i in wide:
                continue
            nonpeak = np.setdiff1d(np.arange(L), wide)
            if np.any(sgn * corr[nonpeak] > sd_k * sd):
                continue
            if np.any(ptail(r[antic], b[antic]) < a_rej):
                continue
            rows.append(dict(pre=int(S['ids'][i]), post=int(S['ids'][j]),
                             i=i, j=j, lag_ms=lags[k] * 1000,
                             z=(r[k] - b[k]) / max(sd, 1e-9), p=p_peak,
                             width_ms=len(wide), n_pre=int(S['n_spikes'][i]),
                             n_post=int(S['n_spikes'][j]), kind=kind,
                             ccg_sum=float(r.sum())))
    return pd.DataFrame(rows)


def n_tested(S):
    r = S['raw'].sum(axis=2)
    np.fill_diagonal(r, 0)
    return int((r >= MIN_CCG).sum())


def sessions_with_labels():
    """Sessions carrying anchoring labels -- the only ones this can speak to."""
    d = ('/Users/harryclark/Documents/spatial-manifolds/data/population_state/labels')
    return sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(d) if f.endswith('.npz')})


def run_all(worker=0, n_workers=1, null_every=4):
    """Detect in every labelled session; jitter null in every `null_every`-th.

    The null costs as much as the detection itself, and its job is to estimate a
    false-positive RATE, which a subset of sessions does as well as all of them.
    Running it everywhere would double a run that already takes hours.
    """
    sess = [s for i, s in enumerate(sessions_with_labels())
            if i % n_workers == worker]
    cpath = f'{OUT}/connections2_w{worker}.csv'
    upath = f'{OUT}/cells2_w{worker}.csv'
    done = set()
    C, U = [], []
    if os.path.exists(cpath):
        C.append(pd.read_csv(cpath)); U.append(pd.read_csv(upath))
        done = set(map(tuple, U[0][['mouse', 'day']].drop_duplicates().values))
    print(f'worker {worker}/{n_workers}: {len(sess)} sessions, {len(done)} done',
          flush=True)
    t0 = time.time()
    for mo, dy in sess:
        if (mo, dy) in done:
            continue
        try:
            S = session_ccgs(mo, dy)
            if S is None:
                print(f'  - M{mo}D{dy}: too few usable cells', flush=True); continue
            # both signs on the same CCGs: the correlograms are the expensive
            # part, and inhibitory connections are the ones the interneuron
            # question needs
            D = pd.concat([detect(S, kind=k) for k in ('exc', 'inh')],
                          ignore_index=True)
            nt = n_tested(S)
            nnull = nnull_inh = np.nan
            if null_every and (len(done) + len(U) - 1) % null_every == 0:
                J = session_ccgs(mo, dy, jitter=True,
                                 rng=np.random.default_rng(abs(hash((mo, dy))) % 2**32))
                if J is not None:
                    nnull = len(detect(J, kind='exc'))
                    nnull_inh = len(detect(J, kind='inh'))
            D['mouse'], D['day'] = mo, dy
            md = S['meta']
            cells = pd.DataFrame(dict(
                mouse=mo, day=dy, cluster_id=S['ids'], n_spikes=S['n_spikes'],
                region=md['brain_region'].values.astype(str) if md is not None else '',
                px=md['coord_probe_x'].values if md is not None else np.nan,
                py=md['coord_probe_y'].values if md is not None else np.nan,
                n_tested=nt, n_conn=len(D), n_null=nnull, n_null_inh=nnull_inh,
                n_exc=int((D.kind == 'exc').sum()) if len(D) else 0,
                n_inh=int((D.kind == 'inh').sum()) if len(D) else 0))
            C.append(D); U.append(cells)
            pd.concat(C, ignore_index=True).to_csv(cpath, index=False)
            pd.concat(U, ignore_index=True).to_csv(upath, index=False)
            print(f'  M{mo}D{dy}: {len(S["ids"])} cells, {nt} pairs, '
                  f'{int((D.kind=="exc").sum())} exc / {int((D.kind=="inh").sum())} inh, '
                  f'null {nnull}/{nnull_inh}, {time.time()-t0:.0f}s', flush=True)
        except Exception as e:
            import traceback
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc(limit=3)
    print(f'worker {worker} done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['batch']:
        w = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        nw = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        run_all(w, nw); sys.exit()
    mo, dy = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (25, 25)
    t0 = time.time()
    S = session_ccgs(mo, dy)
    print(f'M{mo}D{dy}: {len(S["ids"])} cells with >= {MIN_SPIKES} spikes, '
          f'{n_tested(S)} ordered pairs tested ({time.time()-t0:.0f}s)')
    for corr_ in (False, True):
        D = detect(S, corrected=corr_)
        tag = 'CORRECTED' if corr_ else 'uncorrected (as in the notebook)'
        print(f'  {tag:34s}: {len(D)} connections '
              f'({100*len(D)/max(n_tested(S),1):.3f}% of tested pairs)')
    t1 = time.time()
    J = session_ccgs(mo, dy, jitter=True, rng=np.random.default_rng(0))
    DJ = detect(J, corrected=True)
    print(f'  JITTER NULL (+-{JITTER_MS} ms)            : {len(DJ)} connections '
          f'({100*len(DJ)/max(n_tested(J),1):.3f}%) [{time.time()-t1:.0f}s]')
    D = detect(S, corrected=True)
    if len(D):
        print('\ntop connections:')
        print(D.sort_values('z', ascending=False)
              .head(10)[['pre', 'post', 'lag_ms', 'z', 'width_ms']]
              .to_string(index=False))

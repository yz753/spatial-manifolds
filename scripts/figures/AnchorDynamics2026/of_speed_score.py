"""Open-field speed score per cell, computed rather than read off.

Usage:  python3 of_speed_score.py [worker n_workers]

WHY NOT USE THE SUPPLIED GLM SCORES. Each open-field session ships a
time-shifted speed GLM (`time_shifted_S_glms.parquet`), and at first glance that
is the obvious source. It is not usable for this question: its `median_score` is
negative for almost every cell (max 0.028 over a whole session) and only about
1% of entorhinal cells reach FDR significance. Whatever that statistic is
measuring, it has almost no dynamic range here, and a null result against a flat
measure says nothing -- it cannot distinguish "these cells are not speed
modulated" from "this score does not detect speed modulation".

So the speed score is computed directly, in the standard form (Kropff et al.
2015): the Pearson correlation between a cell's instantaneous firing rate and
running speed, in time bins, over moving periods.

    score = r(rate, speed)

tested against a circular-shift null. Shifting preserves both the firing rate
distribution and its autocorrelation while destroying the temporal relationship
to speed, which a shuffle of independent bins would not -- and firing is
strongly autocorrelated at these bin widths, so an uncorrected parametric p
would be badly anti-conservative.

Both OF1 and OF2 are scored where present, and the two are reported separately
rather than averaged: agreement between them is the honest reliability estimate
for a per-cell score, and it is printed at the end.

Writes data/population_state/of_speed_score.csv.
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
from paper_units import curate

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
SRC = '/Volumes/datasetsSD/Ephys_data/clark2025'
OUT = f'{ROOT}/data/population_state/of_speed_score.csv'
BIN = 0.2              # s
SMOOTH = 2             # bins, boxcar on both rate and speed
MIN_SPEED = 2.0        # cm/s, moving only
MIN_BINS = 300
N_SHIFT = 200
MIN_SPIKES = 100


def session(mo, dy, of, rng):
    bp = f'{SRC}/M{mo}/D{dy}/{of}/sub-M{mo}_ses-D{dy}_typ-{of}_beh.nwb'
    cp = (f'{SRC}/M{mo}/D{dy}/{of}/sub-M{mo}_ses-D{dy}_typ-{of}'
          '_srt-kilosort4_clusters.npz')
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    beh = nap.load_file(bp)
    cur = curate(nap.load_file(cp), mo, dy)
    ids = [int(c) for c in cur.index]
    if not ids:
        return None
    S = beh['S']
    ts, vs = np.asarray(S.index), np.asarray(S.values, dtype=float)
    edges = np.arange(ts[0], ts[-1], BIN)
    if len(edges) < MIN_BINS:
        return None
    ctr = edges[:-1]
    j = np.searchsorted(edges, ts) - 1
    ok = (j >= 0) & (j < len(ctr)) & np.isfinite(vs)
    cnt = np.bincount(j[ok], minlength=len(ctr)).astype(float)
    spd = np.where(cnt > 0, np.bincount(j[ok], weights=vs[ok],
                                        minlength=len(ctr)) / np.maximum(cnt, 1),
                   np.nan)
    box = np.ones(SMOOTH) / SMOOTH
    spd = np.convolve(pd.Series(spd).ffill().bfill().values, box, mode='same')
    keep = spd >= MIN_SPEED
    if keep.sum() < MIN_BINS:
        return None
    rows = []
    for c in ids:
        t = np.asarray(cur[c].index, dtype=float)
        if len(t) < MIN_SPIKES:
            continue
        rate = np.convolve(np.histogram(t, bins=edges)[0] / BIN, box, mode='same')
        x, y = spd[keep], rate[keep]
        if np.std(y) == 0 or np.std(x) == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        # circular shift of the FULL rate trace, then re-select, so the shift
        # cannot be undone by the moving mask
        null = np.empty(N_SHIFT)
        full = np.convolve(np.histogram(t, bins=edges)[0] / BIN, box, mode='same')
        for k, s_ in enumerate(rng.integers(MIN_BINS, len(full) - MIN_BINS,
                                            N_SHIFT)):
            yk = np.roll(full, int(s_))[keep]
            null[k] = np.corrcoef(x, yk)[0, 1]
        rows.append(dict(mouse=mo, day=dy, of=of, cluster_id=c,
                         speed_r=r, speed_p=float(np.mean(np.abs(null) >= abs(r))),
                         speed_null=float(np.mean(np.abs(null))),
                         n_bins=int(keep.sum()), n_spikes=len(t),
                         mean_rate=float(np.mean(y))))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    d = f'{ROOT}/data/population_state/labels'
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(d) if f.endswith('.npz')})
    mine = [s for i, s in enumerate(sess) if i % nw == w]
    path = OUT if nw == 1 else OUT.replace('.csv', f'_w{w}.csv')
    out, t0 = [], time.time()
    if os.path.exists(path):
        out.append(pd.read_csv(path))
    done = (set(map(tuple, out[0][['mouse', 'day']].drop_duplicates().values))
            if out else set())
    print(f'worker {w}/{nw}: {len(mine)} sessions, {len(done)} done', flush=True)
    for mo, dy in mine:
        if (mo, dy) in done:
            continue
        got = []
        for of in ('OF1', 'OF2'):
            try:
                r = session(mo, dy, of,
                            np.random.default_rng(abs(hash((mo, dy, of))) % 2**32))
            except Exception as e:
                print(f'  ! M{mo}D{dy} {of}: {type(e).__name__}: {e}', flush=True)
                continue
            if r is not None and len(r):
                got.append(r)
        if not got:
            continue
        out.append(pd.concat(got, ignore_index=True))
        pd.concat(out, ignore_index=True).to_csv(path, index=False)
        print(f'  M{mo}D{dy}: {sum(len(g) for g in got)} cells, '
              f'{time.time() - t0:.0f}s', flush=True)
    A = pd.concat(out, ignore_index=True)
    print(f'\n{len(A)} cell-sessions, {A.groupby(["mouse", "day"]).ngroups} sessions')
    print(f'  speed r: median {A.speed_r.median():+.3f}, '
          f'{100 * (A.speed_p < .05).mean():.0f}% beat their shift null')
    p = A.pivot_table(index=['mouse', 'day', 'cluster_id'], columns='of',
                      values='speed_r')
    if {'OF1', 'OF2'}.issubset(p.columns):
        q = p.dropna()
        print(f'  OF1 vs OF2 agreement: r = {np.corrcoef(q.OF1, q.OF2)[0, 1]:.3f} '
              f'({len(q)} cells in both)')

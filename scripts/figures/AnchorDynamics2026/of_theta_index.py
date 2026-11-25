"""Open-field theta index per cell, computed rather than read off.

Usage:  python3 of_theta_index.py

WHY IT IS COMPUTED HERE. detect_grids.py reads a theta_index.parquet out of
each session's tuning_scores folder, but no such file exists in this dataset --
the reference is to something that was never generated. So the index is built
directly, and from the OPEN FIELD rather than the virtual track, for the same
reason grid score and speed score are: a measure taken on the session being
explained cannot be used to explain it.

THE DEFINITION, following Kornienko et al.

    1. spike times to an instantaneous firing rate: counts in 2 ms bins,
       smoothed with a Gaussian of sigma 5 ms
    2. power spectrum by Welch's method, 2048-sample segments at 500 Hz with
       half overlap, which resolves 0.24 Hz -- fine enough to separate the
       theta band from the flanks it is compared against
    3. theta index

           T = (q - baseline) / (q + baseline)

       where q is mean power in 6-10 Hz and baseline is mean power across the
       two adjacent bands, 3-5 and 11-13 Hz.

A contrast between a band and its own flanks, so it is bounded in [-1, +1],
dimensionless, and insensitive to firing rate -- which matters because the
comparison runs across classes whose rates differ several-fold. Positive means
more power in theta than beside it.

AN EARLIER VERSION USED A DIFFERENT INDEX: the autocorrelogram's power
spectrum, scored as peak power in 5-11 Hz over mean power in 0-50 Hz. That is
also a standard form, but it is a peak-to-broadband ratio rather than a
contrast against the immediately adjacent bands, it is unbounded, and a cell
with a wide low-frequency shoulder scores highly on it without being
theta-rhythmic. The band-contrast form is the one used here.

Both OF1 and OF2 are scored where present and reported separately, since
agreement between them is the honest reliability estimate for a per-cell score.

WHAT IT FOUND. Scored independently in the two open-field sessions of a day,
the index agrees at Spearman rho = 0.865 over 11,810 cells. It separates the
identities -- median 0.229 in putative interneurons, 0.088 in grid cells,
0.046 and 0.020 in the two remaining classes -- and over 4,890 entorhinal
cells it predicts agreement with the population axis: +0.173 per unit index
(p = 1.2e-11) on its own and +0.136 (p = 1.9e-06) with identity in the model,
so most of the relationship is NOT identity. It also holds within classes
taken separately: grid +0.353 (p = 0.037), non-grid spatial +0.147
(p = 8.6e-05), putative interneuron +0.116 (p = 0.028), non-spatial not
significant.

The band-contrast index answers this question where the autocorrelogram form
did not. Under the earlier peak-over-broadband version the within-class
relationship vanished -- grid cells gave rho = -0.000 -- and the pooled effect
looked like identity in disguise. The two indices are measuring different
things, the band-contrast one is the more reliable of them (rho 0.865 against
0.746 between open-field sessions), and the conclusion is the opposite.

Writes data/population_state/of_theta_index.csv
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from paper_units import curate

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
SRC = '/Volumes/datasetsSD/Ephys_data/clark2025'
OUT = f'{ROOT}/data/population_state/of_theta_index.csv'
RATE_BIN = 0.002        # s, 500 Hz
SMOOTH_SD = 0.005       # s, Gaussian on the rate
NPERSEG = 2048          # Welch segment, ~4.1 s, resolves 0.24 Hz
THETA = (6.0, 10.0)
FLANKS = ((3.0, 5.0), (11.0, 13.0))
MIN_SPIKES = 200
MIN_DUR = 120.0         # s of recording, or Welch has too few segments


def theta_index(t):
    """T = (theta - flanks) / (theta + flanks) on the rate's Welch spectrum."""
    t = np.asarray(t, float)
    t0, t1 = t.min(), t.max()
    if t1 - t0 < MIN_DUR:
        return np.nan, np.nan
    edges = np.arange(t0, t1 + RATE_BIN, RATE_BIN)
    r = np.histogram(t, bins=edges)[0] / RATE_BIN
    r = gaussian_filter1d(r.astype(float), SMOOTH_SD / RATE_BIN)
    f, P = welch(r, fs=1.0 / RATE_BIN, nperseg=min(NPERSEG, len(r)),
                 noverlap=min(NPERSEG, len(r)) // 2)
    th = (f >= THETA[0]) & (f <= THETA[1])
    fl = np.zeros_like(th)
    for lo, hi in FLANKS:
        fl |= (f >= lo) & (f <= hi)
    if not th.any() or not fl.any():
        return np.nan, np.nan
    q, b = float(P[th].mean()), float(P[fl].mean())
    if q + b <= 0:
        return np.nan, np.nan
    return (q - b) / (q + b), float(f[th][np.argmax(P[th])])


def session(mo, dy, of):
    cp = (f'{SRC}/M{mo}/D{dy}/{of}/sub-M{mo}_ses-D{dy}_typ-{of}'
          '_srt-kilosort4_clusters.npz')
    if not os.path.exists(cp):
        return []
    cur = curate(nap.load_file(cp), mo, dy)
    rows = []
    for c in [int(x) for x in cur.index]:
        t = np.asarray(cur[c].index, dtype=float)
        if len(t) < MIN_SPIKES:
            continue
        ti, fpk = theta_index(t)
        rows.append(dict(mouse=mo, day=dy, cluster_id=c, of=of,
                         n_spikes=len(t), theta_index=ti, theta_peak_hz=fpk))
    return rows


if __name__ == '__main__':
    import glob
    # parse from the path components, not by splitting on '/D' -- the source
    # root is .../Downloads/..., which contains one
    sess = sorted({(int(_p.split(os.sep)[-3][1:]),
                    int(_p.split(os.sep)[-2][1:]))
                   for _p in glob.glob(f'{SRC}/M*/D*/OF*')})
    out, t0 = [], time.time()
    for k, (mo, dy) in enumerate(sess, 1):
        for of in ('OF1', 'OF2'):
            r = session(mo, dy, of)
            out += r
            if r:
                print(f'[{k}/{len(sess)}] M{mo}D{dy} {of}: {len(r)} cells '
                      f'({time.time() - t0:.0f}s)', flush=True)
    D = pd.DataFrame(out)
    D.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}: {len(D)} rows')
    if len(D):
        print(D.groupby('of').theta_index.describe()[['count', '50%', 'max']])

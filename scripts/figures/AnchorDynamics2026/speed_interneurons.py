"""Do OPEN-FIELD-identified speed interneurons change their VR speed coding
between anchoring states?

THE HYPOTHESIS. Entorhinal speed cells are largely fast-spiking inhibitory PV+
neurons (Ye et al. 2018) that inherit a speed signal from the medial septum
(Justus et al. 2017), which in turn receives it from the mesencephalic locomotor
region. If the anchored state reflects a change in that septal SPEED pathway --
as opposed to the septal PACEMAKER, which would predict disrupted theta timing
instead -- then the cells carrying the pathway should be the cells whose speed
coding changes with state.

WHY IDENTITY COMES FROM THE OPEN FIELD. Defining speed cells inside the VR
session, pooled across states, is what speed_tuning.py does and is the right
choice for the question IT asks. It is the wrong choice here: this analysis asks
whether a specific afferent population changes, so that population has to be
defined by something the VR task cannot have shaped. Waveform class and the
open-field speed score are both measured outside the session being analysed, on
a different environment and a different day's behaviour.

FOUR GROUPS, so that any effect can be shown to be specific rather than general:
    speed interneuron      narrow + high-rate, OF speed-modulated  <- the target
    non-speed interneuron  narrow + high-rate, not OF speed-modulated
    speed principal        broad, OF speed-modulated
    non-speed principal    broad, not OF speed-modulated

MEASURES, per cell and per state, on 250 ms running bins (>= 3 cm/s):
    slope   OLS rate on speed (Hz per cm/s) -- primary, insensitive to the
            narrower speed range of anchored epochs
    gain    slope / mean rate -- separates a change in speed SENSITIVITY from a
            change in overall firing rate
    r       Pearson correlation, reported but not relied on

Each is computed twice: on all running bins, and after trimming both states to
their common 5-95% speed support, because anchored epochs have a materially
narrower speed distribution (SD 6.07 vs 8.58 cm/s) and an untrimmed correlation
would be attenuated by that alone.

Writes data/population_state/speed_interneurons.csv
"""
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
SOURCE = '/Volumes/datasetsSD/Ephys_data/clark2025'
OUT = f'{ROOT}/data/population_state/speed_interneurons.csv'

BIN_S = 0.250
RUN_SPEED = 3.0
MIN_BINS = 40            # per state, per cell
TRIM = (5, 95)


def vr_paths(mo, dy):
    d = f'{SOURCE}/M{mo}/D{dy}/VR/'
    s = f'sub-M{mo}_ses-D{dy}_typ-VR'
    return d + s + '_beh.nwb', d + s + '_srt-kilosort4_clusters.npz'


def fit(rate, spd):
    """slope (Hz per cm/s), gain (slope / mean rate), pearson r."""
    if len(rate) < MIN_BINS or np.std(spd) == 0:
        return np.nan, np.nan, np.nan
    b = np.polyfit(spd, rate, 1)[0]
    mr = rate.mean()
    g = b / mr if mr > 0 else np.nan
    r = np.corrcoef(spd, rate)[0, 1] if np.std(rate) > 0 else np.nan
    return b, g, r


def run_session(mo, dy, UT, state):
    bp, cp = vr_paths(mo, dy)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return []
    u = UT[(UT.mouse == mo) & (UT.day == dy)]
    if u.empty:
        return []
    beh = nap.load_file(bp)
    clusters = nap.load_file(cp)
    ids = [int(c) for c in u.cluster_id.astype(int) if int(c) in clusters.index]
    if not ids:
        return []

    S, tn = beh['S'], beh['trial_number']
    t1 = float(np.asarray(S.index)[-1])
    edges = np.arange(0, t1, BIN_S)
    ctr = edges[:-1] + BIN_S / 2
    st_ = np.asarray(S.index); sv = np.asarray(S.values, float)
    spd = np.interp(ctr, st_, sv)
    tt_ = np.asarray(tn.index); tv = np.asarray(tn.values)
    tr = tv[np.searchsorted(tt_, ctr).clip(0, len(tv) - 1)].astype(int)
    lab = np.array([state.get((mo, dy, int(x)), np.nan) for x in tr], float)
    ok = (spd >= RUN_SPEED) & np.isfinite(lab)
    if ok.sum() < 4 * MIN_BINS:
        return []
    anch = ok & (lab == 1)
    non = ok & (lab == 0)
    if anch.sum() < MIN_BINS or non.sum() < MIN_BINS:
        return []
    # common speed support, computed once per session
    loa, hia = np.percentile(spd[anch], TRIM)
    lon, hin = np.percentile(spd[non], TRIM)
    lo, hi = max(loa, lon), min(hia, hin)
    trim_a = anch & (spd >= lo) & (spd <= hi)
    trim_n = non & (spd >= lo) & (spd <= hi)

    meta = u.set_index('cluster_id')
    rows = []
    for c in ids:
        sp = np.asarray(clusters[c].index, float)
        cnt = np.histogram(sp, bins=edges)[0].astype(float)
        rate = cnt / BIN_S
        m = meta.loc[c]
        rec = dict(mouse=mo, day=dy, cluster_id=c, group=m.group,
                   n_anch=int(anch.sum()), n_non=int(non.sum()))
        for tag, (ma, mn) in (('all', (anch, non)), ('trim', (trim_a, trim_n))):
            for sname, msk in (('anch', ma), ('non', mn)):
                b, g, r = fit(rate[msk], spd[msk])
                rec[f'{tag}_{sname}_slope'] = b
                rec[f'{tag}_{sname}_gain'] = g
                rec[f'{tag}_{sname}_r'] = r
                rec[f'{tag}_{sname}_rate'] = rate[msk].mean()
        rows.append(rec)
    return rows


def main():
    U = pd.read_csv(f'{ROOT}/data/population_state/unit_table.csv')
    U = U[U.brain_region.astype(str).str.startswith('ENTm')].copy()
    U['speed_cell'] = (U.speed_p < 0.05) & (U.speed_r > 0)
    U = U.dropna(subset=['speed_p', 'speed_r', 'putative_int'])
    U['group'] = np.where(U.putative_int,
                          np.where(U.speed_cell, 'speed interneuron',
                                   'non-speed interneuron'),
                          np.where(U.speed_cell, 'speed principal',
                                   'non-speed principal'))
    print('cells by group (open-field defined):')
    print(U.group.value_counts().to_string(), '\n')

    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    state = {(int(r.mouse), int(r.day), int(r.trial)): float(r.frac_anch > .5)
             for r in T.itertuples()}

    sess = sorted({(int(a), int(b)) for a, b in
                   U[['mouse', 'day']].drop_duplicates().values})
    rows = []
    for k, (mo, dy) in enumerate(sess, 1):
        try:
            r = run_session(mo, dy, U, state)
        except Exception as e:
            print(f'[{k}/{len(sess)}] M{mo}D{dy} ! {type(e).__name__}: {e}', flush=True)
            continue
        rows += r
        print(f'[{k}/{len(sess)}] M{mo}D{dy} {len(r)} cells', flush=True)
    D = pd.DataFrame(rows)
    if D.empty:
        sys.exit('no rows')
    D.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}: {len(D)} cells, '
          f'{D.drop_duplicates(["mouse","day"]).shape[0]} sessions')


if __name__ == '__main__':
    main()

"""One row per cell: who it is, whether it follows the population, and why.

Usage:  python3 build_unit_table.py [--rebuild]

Figure 2 asks three questions about single units, and each needs a different
source joined onto the same cell:

  WHICH IDENTITIES FOLLOW?   open-field class (grid / non-grid spatial /
                             non-spatial) against the null-corrected agreement
                             with the population anchoring axis
  WHAT DO INTERNEURONS DO?   there is no interneuron label in this dataset, so
                             one is derived from the spike waveform
  WHO DISSENTS, AND WHY?     cells that stay anchored while the population is
                             not, tested against open-field speed modulation

Writes data/population_state/unit_table.csv.

FOUR SOURCES, JOINED ON (mouse, day, cluster_id)

  per_cell_pc1.csv        agreement r with the population axis, recomputed
                          leave-one-out so a cell is not scored against a
                          component it helped define, plus its circular-shift
                          null. `excess = r - r_null` is the comparable quantity.
  labels/M*D*.npz         the per-trial labels themselves, for the dissent
                          measures below, which cannot be read off r
  cell_classifications_v2 open-field identity and anatomy
  clusters .npz metadata  spike waveform, for the interneuron call
  of_speed_score.csv      open-field speed score, COMPUTED (of_speed_score.py),
                          not the supplied time-shifted GLM -- that statistic is
                          negative for almost every cell and reaches
                          significance in ~1% of entorhinal cells, so it has no
                          dynamic range to test anything against. The computed
                          score beats its shift null in ~60% of cells and
                          replicates across OF1/OF2 at r = 0.75.

WHY A DISSENT MEASURE AND NOT JUST r. A cell can disagree with the population in
two different ways and r does not distinguish them: it can be anchored when the
population is not (a dissenter that holds its map), or non-anchored when the
population is (a cell that drops out). Both give a low r. So:

    stay  = P(cell anchored | population NON-anchored)
    drop  = P(cell non-anchored | population anchored)

`stay` is the one the speed hypothesis is about -- a cell that keeps a stable
track map while the population loses one. The two are reported separately
because a single symmetric score would average them into nothing.

THE INTERNEURON CALL IS A GUESS AND IS LABELLED AS ONE. No ground truth exists
here, so putative fast-spiking cells are taken as narrow-waveform AND
high-rate. The threshold is not assumed: `interneuron_threshold()` fits a
two-component Gaussian mixture to log peak-to-trough duration and takes the
crossing point, and the script prints the separation so a reader can see whether
the distribution was bimodal at all. A unimodal distribution would mean the
split is arbitrary, and that has to be visible rather than buried.

OPEN-FIELD SPEED is r(firing rate, running speed) in 200 ms bins over moving
periods, against a circular-shift null. OF1 is preferred where both open-field
sessions exist, with OF2 as the reliability check rather than something to
average in.
"""
import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from spatial_manifolds.anchoring import load_session_labels
from paper_units import curate

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
SRC = '/Volumes/datasetsSD/Ephys_data/clark2025'
PS = f'{ROOT}/data/population_state'
OUT = f'{PS}/unit_table.csv'
WAVE = ['peak_to_trough_duration', 'trough_half_width', 'peak_half_width',
        'firing_rate', 'spread']
MIN_STATE_TRIALS = 10      # per state, for a session to count as switching


def session_labels_measures(mo, dy):
    """Per-cell dissent measures from the per-trial label matrix."""
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    L, ids = z['labels'], z['cluster_id'].astype(int)
    pop = np.asarray(z['frac_anch'], dtype=float) > .5
    # Whether the POPULATION switched at all. In a session that never leaves one
    # state every cell is "locked" by construction, which says nothing about the
    # cell -- so any analysis of locking has to be able to exclude these.
    n_a, n_n = int(pop.sum()), int((~pop).sum())
    switches = (n_a >= MIN_STATE_TRIALS) and (n_n >= MIN_STATE_TRIALS)
    rows = []
    for i, c in enumerate(ids):
        lab = L[i]
        ok = np.isfinite(lab)
        a_when_pop_off = ok & ~pop
        n_when_pop_on = ok & pop
        rows.append(dict(
            mouse=mo, day=dy, cluster_id=int(c),
            stay=float(np.mean(lab[a_when_pop_off] == 1))
            if a_when_pop_off.sum() >= 5 else np.nan,
            drop=float(np.mean(lab[n_when_pop_on] == 0))
            if n_when_pop_on.sum() >= 5 else np.nan,
            n_pop_off=int(a_when_pop_off.sum()), n_pop_on=int(n_when_pop_on.sum()),
            cell_frac_anch=float(np.nanmean(lab)),
            label_sd=float(np.nanstd(lab)),
            in_population=bool(np.isfinite(z['pc1_load'][i])),
            pc1_load=float(z['pc1_load'][i]),
            sess_n_anch=n_a, sess_n_non=n_n, sess_switches=switches))
    return pd.DataFrame(rows)


def session_waveforms(mo, dy):
    """Waveform metrics from the VR clusters file."""
    p = (f'{SRC}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR'
         '_srt-kilosort4_clusters.npz')
    if not os.path.exists(p):
        return None
    cur = curate(nap.load_file(p), mo, dy)
    md = cur.metadata
    keep = [w for w in WAVE if w in md.columns]
    out = md[keep].copy()
    out.insert(0, 'cluster_id', [int(c) for c in cur.index])
    out.insert(0, 'day', dy); out.insert(0, 'mouse', mo)
    return out.reset_index(drop=True)


def session_speed_glm(mo, dy):
    """DEPRECATED -- the supplied GLM scores are not used. See of_speed_score.py."""
    out = []
    for of in ('OF1', 'OF2'):
        p = (f'{SRC}/M{mo}/D{dy}/{of}/tuning_scores/kilosort4/'
             'time_shifted_S_glms.parquet')
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        # `d.shift` is the DataFrame method, not the column -- it must be
        # subscripted, or every session silently raises and is skipped.
        g = d.groupby('cluster_id')
        z = (d.loc[d['shift'].abs().groupby(d.cluster_id).idxmin()]
             .set_index('cluster_id'))
        r = pd.DataFrame(dict(
            cluster_id=z.index.astype(int),
            speed_score=z.median_score.values,
            speed_p=z.p_val_fdr.values,
            speed_score_peak=g.median_score.max().reindex(z.index).values,
            speed_p_best=g.p_val_fdr.min().reindex(z.index).values))
        r['of'] = of
        out.append(r)
    if not out:
        return None
    # OF1 is the session used for the classification datasheet; prefer it
    A = pd.concat(out)
    A = A.sort_values('of').groupby('cluster_id').first().reset_index()
    A.insert(0, 'day', dy); A.insert(0, 'mouse', mo)
    return A


PTT_THRESHOLD = 0.4e-3     # seconds; the narrow/broad boundary actually used
# Below this the template metric failed rather than measured a spike: 43 units
# read 0.01 ms, a third of one 30 kHz sample. They are left unmeasured (NaN),
# so they keep their open-field class; with them, the mixture's separation
# falls from 3.9 to 1.6 SD on the narrow side's log tail alone.
PTT_MIN = 0.05e-3


def interneuron_threshold(ptt_s, verbose=True):
    """Antimode of log peak-to-trough duration, from a 2-component mixture.

    NOT the threshold used -- that is the fixed PTT_THRESHOLD of 0.4 ms. This
    function exists to CORROBORATE that choice: it reports where an unsupervised
    two-component fit puts the boundary, and how well separated the components
    are. The fit lands at 0.42 ms, within one histogram bin of 0.4, and the
    empirical density minimum sits at 0.38-0.40 ms, so 0.4 is if anything the
    better cut as well as the conventional one.

    A fitted antimode is also unstable in a way a convention is not: it moves
    with the subset of cells supplied, so sessions, mice and reanalyses would
    each get a slightly different boundary, and cells would change class for
    reasons that have nothing to do with the cells. The separation index is the
    number that matters here -- below about 1.5 the two components are not
    resolved and any threshold would be arbitrary.

    Returned in SECONDS, alongside the separation index.
    """
    from sklearn.mixture import GaussianMixture
    v = np.log(ptt_s[np.isfinite(ptt_s) & (ptt_s > 0)]).reshape(-1, 1)
    gm = GaussianMixture(2, n_init=10, random_state=0).fit(v)
    mu = gm.means_.ravel(); sd = np.sqrt(gm.covariances_.ravel())
    order = np.argsort(mu)
    mu, sd = mu[order], sd[order]
    grid = np.linspace(mu[0], mu[1], 2000).reshape(-1, 1)
    thr = float(np.exp(grid[np.argmin(gm.score_samples(grid))][0]))
    sep = float((mu[1] - mu[0]) / np.sqrt(np.mean(sd ** 2)))
    if verbose:
        print(f'  waveform mixture: means {np.exp(mu) * 1000} ms, '
              f'separation {sep:.2f} SD, antimode {thr * 1000:.3f} ms '
              f'(corroborating; {PTT_THRESHOLD * 1000:.1f} ms is used)')
    return thr, sep


def build():
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    lab, wav, spd = [], [], []
    t0 = time.time()
    for mo, dy in sess:
        for fn, acc in ((session_labels_measures, lab), (session_waveforms, wav)):
            try:
                r = fn(mo, dy)
            except Exception as e:
                print(f'  ! M{mo}D{dy} {fn.__name__}: {type(e).__name__}: {e}',
                      flush=True); continue
            if r is not None and len(r):
                acc.append(r)
    L = pd.concat(lab, ignore_index=True)
    W = pd.concat(wav, ignore_index=True)
    # the single-run file when present; worker files only stand in for it
    # (globbing both counted every cell two or three times)
    sp = ([f'{PS}/of_speed_score.csv'] if os.path.exists(f'{PS}/of_speed_score.csv')
          else sorted(glob.glob(f'{PS}/of_speed_score_w*.csv')))
    if not sp:
        raise FileNotFoundError('run of_speed_score.py first')
    S = pd.concat([pd.read_csv(f) for f in sp], ignore_index=True)
    # OF1 preferred; OF2 kept separately as the reliability check
    S2 = S[S.of == 'OF2'][['mouse', 'day', 'cluster_id', 'speed_r']].rename(
        columns={'speed_r': 'speed_r_of2'})
    S = (S[S.of == 'OF1'][['mouse', 'day', 'cluster_id', 'speed_r', 'speed_p',
                           'speed_null', 'mean_rate']]
         .merge(S2, on=['mouse', 'day', 'cluster_id'], how='outer'))
    print(f'labels {len(L)} cells, waveforms {len(W)}, OF speed {len(S)} '
          f'({time.time() - t0:.0f}s)')

    P = pd.read_csv(f'{PS}/per_cell_pc1.csv')[
        ['mouse', 'day', 'cluster_id', 'r', 'p', 'cls']]
    C = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')[
        ['mouse', 'day', 'cluster_id', 'cell_class_of1', 'brain_region',
         'grid_score_of1', 'spatial_information_of1', 'mean_rate_of1',
         'hd_mean_vector_length_of1', 'coord_probe_x', 'coord_probe_y']]
    key = ['mouse', 'day', 'cluster_id']
    U = (L.merge(P, on=key, how='left').merge(C, on=key, how='left')
         .merge(W, on=key, how='left').merge(S, on=key, how='left'))

    # The mixture is fitted for its separation index and its antimode, both of
    # which are reported, but the boundary applied is the fixed 0.4 ms -- see
    # interneuron_threshold for why a convention beats a refitted cut here.
    U.loc[U.peak_to_trough_duration < PTT_MIN, 'peak_to_trough_duration'] = np.nan
    antimode, sep = interneuron_threshold(U.peak_to_trough_duration.values)
    U['narrow'] = U.peak_to_trough_duration < PTT_THRESHOLD
    rate_thr = U.firing_rate.quantile(.75)
    U['putative_int'] = U.narrow & (U.firing_rate >= rate_thr)
    U['wave_threshold_s'] = PTT_THRESHOLD
    U['wave_antimode_s'] = antimode
    U['wave_separation_sd'] = sep
    U['rate_threshold_hz'] = rate_thr

    def identity(r):
        if r.putative_int:
            return 'putative interneuron'
        return {'GC': 'grid', 'NGS': 'non-grid spatial', 'NS': 'non-spatial',
                'LOW': 'low rate', 'EXCL': 'excluded'}.get(r.cell_class_of1,
                                                           'unclassified')
    U['identity'] = U.apply(identity, axis=1)
    U.to_csv(OUT, index=False)
    print(f'wrote {OUT}: {len(U)} cells, '
          f'{U.groupby(["mouse", "day"]).ngroups} sessions')
    print('  identity:', U.identity.value_counts().to_dict())
    sw = U.groupby(['mouse', 'day']).sess_switches.first()
    print(f'  switching sessions: {int(sw.sum())} of {len(sw)} '
          f'(>= {MIN_STATE_TRIALS} trials in each state)')
    print(f'  with an OF speed score: {int(U.speed_r.notna().sum())} '
          f'({100 * (U.speed_p < .05).mean():.0f}% beat their shift null)')
    print(f'  with an agreement score r: {int(U.r.notna().sum())}')
    return U


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true')
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.rebuild:
        U = pd.read_csv(OUT)
        print(f'{OUT}: {len(U)} cells (cached; --rebuild to redo)')
    else:
        build()

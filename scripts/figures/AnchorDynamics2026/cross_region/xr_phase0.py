"""Phase 0 for the cross-region coupling analysis: fix the geometry, then time it.

THE SHANK BUG. The notebook assigned shanks with
    pd.qcut(coord_probe_x.rank(), 4) if nunique > 3 else 0
Neuropixels 2.0 places four shanks at x = 0, 250, 500, 750 um, each with two
columns 32 um apart, so x takes 8 values when all four shanks carry cells and 2
when only one does. The qcut branch therefore split ONE shank's two columns into
four fake "shanks", and the else branch gave every cell shank 0 -- which makes
`cells.shank != target.shank` exclude everything, leaving the within-region pool
empty and the `within` ceiling silently NaN. Since the whole framing is "where
does cross sit between the shift floor and the within ceiling", that ceiling
going missing is not a cosmetic bug.

    shank = round(coord_probe_x / 250)

THE EXCLUSION RULE. Excluding the target's own shank is a proxy for "not
physically adjacent", and it fails in single-shank sessions, where it removes
every candidate. The rule that generalises is distance in probe space: a
within-region pool cell must be on a different shank OR more than MIN_UM away
along the shank. Same intent, defined in the units the intent is about.

Also fixed here:
  n_filters 100 -> 5              (was one filter per bin, not a basis)
  position -> sin/cos             (the track is a ring; 0 and 200 cm are one place)
  baseline fitted once per target  (it does not depend on the pool, and was being
                                    refitted for every draw of every condition --
                                    15 wasted fits per target)
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from spatial_manifolds.mlencoding import MLencoding, poisson_pseudoR2

SOURCE_PATH = '/Volumes/datasetsSD/Ephys_data/clark2025/'
TIME_BS, HISTORY_MS, TL = 10, 1000, 200.0
N_FILTERS = 5                  # raised-cosine basis, NOT one filter per bin
N_POOL, MIN_SPIKES = 10, 500
MIN_UM = 100.0                 # within-region pool must be this far from the target
CURATION = {'good', 'mua'}

CEREB = ('SIM', 'CUL', 'CENT', 'AN', 'PRM', 'COPY', 'PFL', 'FL', 'DEC', 'FOTU',
         'PYR', 'UVU', 'NOD', 'CB')


def region_group(r):
    r = str(r)
    if r.startswith('ENTm'): return 'MEC'
    if r.startswith('ENTl'): return 'LEC'
    if r.startswith('VIS'):  return 'VIS'
    if r.startswith(('PRE', 'POST', 'PAR', 'SUB')): return 'SUB'
    if r.startswith(CEREB):  return 'CB'
    return 'OTHER'


def session_cells(mouse, day):
    d = f'{SOURCE_PATH}M{mouse}/D{day}/VR/'
    stem = f'sub-M{mouse}_ses-D{day}_typ-VR'
    bp, cp = d + stem + '_beh.nwb', d + stem + '_srt-kilosort4_clusters.npz'
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    z = np.load(cp, allow_pickle=True)
    md = z['_metadata'].item()
    df = pd.DataFrame({k: np.asarray(md[k]) for k in
                       ('brain_region', 'bombcell_label', 'coord_probe_x',
                        'coord_probe_y', 'cell_class_of1')
                       if k in md})
    df['cluster_id'] = np.asarray(z['keys'])
    df['region'] = df.brain_region.map(region_group)
    df['shank'] = (df.coord_probe_x.astype(float) / 250).round().astype(int)
    df['py'] = df.coord_probe_y.astype(float)
    df = df[df.bombcell_label.isin(CURATION)]
    return dict(beh_path=bp, clu_path=cp, cells=df)


def within_pool(cells, t_row):
    """Same region, but not physically adjacent to the target.

    Different shank, or far enough along the same shank. The point is to keep
    spike-sorting bleed between neighbouring contacts out of the ceiling.
    """
    same = cells[(cells.region == t_row.region) & (cells.cluster_id != t_row.cluster_id)]
    far = (same.shank != t_row.shank) | ((same.py - t_row.py).abs() > MIN_UM)
    return same[far].cluster_id.values


def load_session(mouse, day):
    s = session_cells(mouse, day)
    beh = nap.load_file(s['beh_path']); clusters = nap.load_file(s['clu_path'])
    ids = list(clusters.index)
    last = clusters[ids[0]].count(bin_size=TIME_BS, time_units='ms').index[-1]
    ep = nap.IntervalSet(start=0, end=last, time_units='s')
    speed = pd.Series(np.asarray(beh['S'].bin_average(bin_size=TIME_BS,
                                                      time_units='ms', ep=ep))).ffill().bfill().values
    trav = pd.Series(np.asarray(beh['travel'].bin_average(bin_size=TIME_BS,
                                                          time_units='ms', ep=ep))).ffill().bfill().values
    pos = trav % TL
    ang = 2 * np.pi * pos / TL                     # the track is a ring
    counts = {c: np.asarray(clusters[c].count(bin_size=TIME_BS, time_units='ms', ep=ep))
              for c in ids}
    cells = s['cells'][s['cells'].cluster_id.isin(ids)].copy()
    cells['n_spikes'] = cells.cluster_id.map({c: int(counts[c].sum()) for c in ids})
    cells = cells[cells.n_spikes >= MIN_SPIKES]
    base = np.column_stack([np.sin(ang), np.cos(ang), speed])
    return dict(base=base, counts=counts, cells=cells, n=len(pos), speed=speed)


def make_model():
    return MLencoding(tunemodel='xgboost', cov_history=True, spike_history=False,
                      window=TIME_BS, n_filters=N_FILTERS, max_time=HISTORY_MS)


if __name__ == '__main__':
    # --- geometry inventory: how often is a shank exclusion even possible?
    import json
    g = {}
    for c in json.load(open('/Users/harryclark/Documents/spatial-manifolds/scripts/'
                            'figures/AnchorDynamics2026/lick_raster_by_trial_type.ipynb'))['cells']:
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        if src.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in src:
            continue
        exec(compile(src, '<nb>', 'exec'), g)
    CLS = g['CLS']
    rows = []
    for (mo, dy), _ in CLS.groupby(['mouse', 'day']):
        s = session_cells(int(mo), int(dy))
        if s is None:
            continue
        c = s['cells']
        rows.append(dict(mouse=mo, day=dy, n=len(c), shanks=c.shank.nunique(),
                         **{r: int((c.region == r).sum()) for r in
                            ('MEC', 'VIS', 'SUB', 'CB')}))
    I = pd.DataFrame(rows)
    I.to_csv('xr_inventory.csv', index=False)
    print(f'{len(I)} sessions')
    print(f'  shank counts: {I.shanks.value_counts().to_dict()}')
    print(f'  single-shank sessions (distance rule is the only option): '
          f'{(I.shanks == 1).sum()}')
    for r in ('VIS', 'SUB', 'CB'):
        print(f'  MEC>=10 & {r}>=10: {((I.MEC >= 10) & (I[r] >= 10)).sum()} sessions')

    # --- timing on one session
    MO, DY = 25, 23
    t0 = time.time()
    S = load_session(MO, DY)
    print(f'\nM{MO}D{DY} loaded in {time.time()-t0:.0f}s: {S["n"]} bins of {TIME_BS} ms, '
          f'{len(S["cells"])} cells >= {MIN_SPIKES} spikes')
    print('  regions:', S['cells'].region.value_counts().to_dict())
    tgt = S['cells'][S['cells'].region == 'MEC'].cluster_id.iloc[0]
    t_row = S['cells'][S['cells'].cluster_id == tgt].iloc[0]
    wp = within_pool(S['cells'], t_row)
    print(f'  target {tgt} on shank {t_row.shank}: within-pool {len(wp)} cells '
          f'(old rule would give {(S["cells"].shank != t_row.shank).sum()})')

    m = make_model()
    y = S['counts'][tgt]
    t0 = time.time()
    _, pr2_base = m.fit_cv(S['base'], y, verbose=0, continuous_folds=True)
    t_base = time.time() - t0
    rng = np.random.default_rng(0)
    pick = rng.choice(S['cells'][S['cells'].region == 'VIS'].cluster_id.values,
                      min(N_POOL, (S['cells'].region == 'VIS').sum()), replace=False)
    X = np.column_stack([S['base']] + [S['counts'][c] for c in pick])
    t0 = time.time()
    _, pr2_full = m.fit_cv(X, y, verbose=0, continuous_folds=True)
    t_full = time.time() - t0
    print(f'\n  baseline fit: {t_base:.1f}s  pR2 {np.nanmean(pr2_base):+.4f}')
    print(f'  +{len(pick)}-cell pool: {t_full:.1f}s  pR2 {np.nanmean(pr2_full):+.4f}  '
          f'dpR2 {np.nanmean(pr2_full)-np.nanmean(pr2_base):+.4f}')
    print(f'\n  per target (1 baseline + 3 conditions x 3 draws, 5 folds): '
          f'~{(t_base + 9 * t_full)/60:.1f} min')

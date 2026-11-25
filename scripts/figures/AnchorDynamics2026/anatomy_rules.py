"""Rule-based gross anatomy along each shank, from curated cells and LFP spectra.

Every shank enters visual cortex, passes into MEC / parasubiculum, and then
either ends there or leaves cortex into the space above the cerebellum, some
reaching cerebellum. So, per SINGLE-SHANK session (curated cells: open-field
class GC, NGS or NS; LFP spectra from boundary_unsupervised, cached):

  top       the most dorsal curated cell: where the shank enters brain
  boundary  VIS -> MEC/PARA: MEC carries a large block of high power from ~40 to
            300 Hz, topped by a ~60 Hz component, that visual cortex lacks. Per
            channel, the z-scored log power in 40-80, 80-150 and 150-300 Hz
            (smoothed over 3 channels); a channel is in a block if any of the
            three is above the shank average. Among blocks between top and the
            exit that begin at least MIN_MEC_UM above the exit (MEC is never
            thinner), the one with the most power is MEC; the boundary is its
            onset, the shallowest contiguous channel at which power has risen
            ONSET_FRAC of the way from the visual cortex level above it to the
            block's peak. Without such a block the shank shows no VIS crossing
            and everything from top to exit is MEC/PARA (boundary = top).
            Against the atlas border: median offset 194 um, 19 of 32 within
            250 um, none beyond 600 um. The earlier 6-60 Hz rule is kept as
            boundary_6_60 for comparison only.
  exit      leaving cortex. The ventral edge of cortex carries a band of strong
            10-44 Hz power (found without choosing a band: it is one of four
            components of a non-negative factorisation of all single-shank
            sessions' channel spectra, peak 31 Hz). Per channel, the mean of
            the positive z-scored log power at 10-44 Hz, smoothed over 3
            channels and scaled by the 99th percentile over all single-shank
            sessions. Its largest value more than 100 um below the boundary
            marks the edge if it is prominent -- at least EDGE_PROM above the
            minimum 250-500 um above it: shanks with cells below a gap (all of
            them cerebellar) reach 0.43-1.1, while shanks ending inside cortex
            reach at most 0.33. Without a prominent peak the shank ends inside
            cortex, at its deepest curated cell. Channels whose whole spectrum
            stands out from their neighbours (single noisy channels) are first
            replaced by the mean of their neighbours.
  gap_exit  the check from the cells: the first run of >= GAP_UM with no
            curated cell beginning more than 150 um below the boundary (or the
            deepest cell). Cells below the exit are cerebellum.

Multi-shank sessions are not given their own estimate: each (mouse, shank) takes
the mean of its single-shank sessions, applied by the cell's shank (nearest
shank mediolateral position), since the probe is the same chronic implant.

Writes data/population_state/rule_boundaries.csv (per single-shank session),
       data/population_state/rule_shanks.csv (per mouse and shank),
       data/population_state/rule_labels.csv (every curated cell, every session)
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boundary_unsupervised as BU
from scipy.ndimage import uniform_filter1d

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
GAP_UM = 300.
BAND = (6., 60.)
EDGE_BAND = (10., 44.)
BB_SUBBANDS = ((40., 80.), (80., 150.), (150., 300.))
ONSET_FRAC = 0.3
MIN_MEC_UM = 500.
EDGE_PROM = 0.4
ARTEFACT_Z = 1.5
_EDGE_NORM = None
CX = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
# the rules are compared with the ATLAS; brain_region itself may already be relabelled by them
if 'brain_region_atlas' in CX:
    CX['brain_region'] = CX.brain_region_atlas
CUR = CX[CX.cell_class_of1.isin(['GC', 'NGS', 'NS'])].dropna(subset=['coord_SCs_y', 'coord_SCs_x'])
DC = pd.read_csv(f'{ROOT}/data/device_contact_id_annotations.csv')
SH = DC.groupby(['mouse', 'shank_id']).coord_SCs_x.mean().abs()


def shank_of(mo, ml):
    return int(np.argmin([abs(SH[(mo, k)] - abs(ml)) for k in range(4)]))


_U = pd.read_csv(f'{PS}/unit_table.csv')
INT = set(zip(_U[_U.identity == 'putative interneuron'].mouse, _U[_U.identity == 'putative interneuron'].day,
              _U[_U.identity == 'putative interneuron'].cluster_id))


def cell_column(cc):
    """Column for each curated cell: 0 grid, 1 non-grid spatial, 2 non-spatial, 3 putative interneuron."""
    base = cc.cell_class_of1.map({'GC': 0, 'NGS': 1, 'NS': 2}).values
    isint = np.array([(int(a), int(b), int(c)) in INT for a, b, c in zip(cc.mouse, cc.day, cc.cluster_id)])
    return np.where(isint, 3, base)

_RATE = {}


def vr_rates(mo, dy):
    """VR firing rate (Hz) of every cluster, from the cluster file's metadata."""
    if (mo, dy) not in _RATE:
        import pynapple as nap
        p = f'/Volumes/datasetsSD/Ephys_data/clark2025/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_srt-kilosort4_clusters.npz'
        m = nap.load_file(p).metadata
        _RATE[(mo, dy)] = dict(zip(m.index.astype(int), m.firing_rate.astype(float)))
    return _RATE[(mo, dy)]


def rate_size(rates, s_min=1.5, s_max=34., r_max=100.):
    """Marker area linear in firing rate up to r_max Hz: cortical cells (mostly
    < 20 Hz) stay small, and the fast cells below cortex -- cerebellar, with
    a tail to 60-80 Hz -- stand out."""
    r = np.clip(np.asarray(rates, float), 0, r_max)
    return s_min + r / r_max * (s_max - s_min)

def single_shank_sessions():
    c = CUR.assign(ml=CUR.coord_SCs_x.abs())
    span = c.groupby(['mouse', 'day']).ml.agg(lambda v: v.max() - v.min())
    return sorted(span[span < 100].index)


def load_spectra(mo, dy):
    cf = f'{BU.OUTD}/spectra_M{mo}D{dy}.npz'
    if os.path.exists(cf):
        z = np.load(cf, allow_pickle=True)
        return pd.DataFrame(z['C']), z['f'], z['LP']
    if not os.path.exists(f'{BU.LFP}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'):
        return None
    C, f, LP = BU.spectra(mo, dy)
    np.savez_compressed(cf, C=C.to_records(index=False), f=f, LP=LP)
    return C, f, LP


def clean_channels(Z):
    """Replace channels whose spectrum departs from their neighbours' median."""
    Z = Z.copy()
    n = len(Z)
    for i in range(n):
        nb = [j for j in range(max(0, i - 2), min(n, i + 3)) if j != i]
        if len(nb) >= 2 and np.abs(Z[i] - np.median(Z[nb], 0)).mean() > ARTEFACT_Z:
            Z[i] = np.mean(Z[[j for j in (i - 1, i + 1) if 0 <= j < n]], 0)
    return Z


def edge_profile(C, f, LP):
    Z = (LP - LP.mean(0)) / LP.std(0)
    o = np.argsort(C.dv.values)
    Z = clean_channels(Z[o])
    return C.dv.values[o], np.clip(Z[:, (f >= EDGE_BAND[0]) & (f <= EDGE_BAND[1])], 0, None).mean(1)


def edge_norm():
    global _EDGE_NORM
    if _EDGE_NORM is None:
        v = []
        for mo, dy in single_shank_sessions():
            got = load_spectra(mo, dy)
            if got:
                v.append(edge_profile(*got)[1])
        _EDGE_NORM = float(np.percentile(np.concatenate(v), 99))
    return _EDGE_NORM


def session_rule(mo, dy):
    got = load_spectra(mo, dy)
    if got is None:
        return None
    C, f, LP = got
    Z = (LP - LP.mean(0)) / LP.std(0)
    o = np.argsort(C.dv.values); dv = C.dv.values[o]; Z = Z[o]
    prof = uniform_filter1d(Z[:, (f >= BAND[0]) & (f <= BAND[1])].mean(1), 3)
    d = np.sort(CUR[(CUR.mouse == mo) & (CUR.day == dy)].coord_SCs_y.values)
    top = d.min()
    b = np.nan
    for i in range(len(prof) - 2):
        if dv[i] > top and (prof[i:i + 3] > 0).all():
            b = (dv[i - 1] + dv[i]) / 2 if i > 0 else dv[i]; break
    b660 = b

    def exits(b):
        gap_exit = d.max()
        for i in range(1, len(d)):
            if d[i] - d[i - 1] >= GAP_UM and d[i - 1] > b + 150:
                gap_exit = d[i - 1]; break
        ev, ep = edge_profile(C, f, LP)
        es = uniform_filter1d(ep / edge_norm(), 3)
        below = ev > b + 100
        prom, edge = np.nan, np.nan
        if below.any():
            i = np.flatnonzero(below)[np.argmax(es[below])]
            above = es[(ev > ev[i] - 500) & (ev < ev[i] - 250)]
            prom = es[i] - (above.min() if len(above) else 0.)
            edge = ev[i]
        return (edge if prom >= EDGE_PROM else d.max()), gap_exit, edge, prom

    exit_, gap_exit, edge, prom = exits(b660)
    # the MEC block: channels where any part of 40-300 Hz is above the shank
    # average, so the ~60 Hz component that tops the broadband block is part of it
    Zc = clean_channels(Z)
    P = np.array([uniform_filter1d(Zc[:, (f >= lo) & (f <= hi)].mean(1), 3) for lo, hi in BB_SUBBANDS])
    act, pw = P.max(0), P.mean(0)
    runs, cur = [], []
    for i in np.flatnonzero((dv > top) & (dv <= exit_ + 60)):
        if act[i] > 0:
            cur.append(i)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    # MEC/PARA below the boundary is at least MIN_MEC_UM thick, so a run that
    # starts closer than that to the exit cannot mark a VIS crossing
    runs = [r_ for r_ in runs if exit_ - dv[r_[0]] >= MIN_MEC_UM]
    if runs:
        best = max(runs, key=lambda r_: np.clip(pw[r_], 0, None).sum())
        j = best[0]
        # up to the component's onset: ONSET_FRAC of the way from the visual
        # cortex level above to the block's peak
        vis = np.flatnonzero((dv >= top) & (dv < dv[best[0]]))
        if len(vis):
            thr = np.median(act[vis]) + ONSET_FRAC * (act[best].max() - np.median(act[vis]))
            while j > 0 and dv[j - 1] >= top and act[j - 1] > thr:
                j -= 1
        b = (dv[j - 1] + dv[j]) / 2 if j > 0 else dv[j]
    else:
        b = top          # no VIS crossing: the whole shank in cortex is MEC/PARA
    exit_, gap_exit, edge, prom = exits(b)
    cb = int((d > exit_ + 30).sum())
    return dict(mouse=mo, day=dy, shank=shank_of(mo, np.median(CUR[(CUR.mouse == mo) & (CUR.day == dy)].coord_SCs_x)),
                top=top, boundary=b, exit=exit_, n_cerebellum=cb, deepest=d.max(),
                gap_exit=gap_exit, edge_peak=edge, edge_prominence=prom, boundary_6_60=b660)


def atlas_region(a):
    a = str(a)
    if a.startswith('VIS'):
        return 'VIS'
    if a.startswith(('ENTm', 'PAR', 'PRE', 'POST')):
        return 'MEC/PARA'
    return 'other'


if __name__ == '__main__':
    R = pd.DataFrame([r for r in (session_rule(mo, dy) for mo, dy in single_shank_sessions()) if r])
    att = []
    for _, r in R.iterrows():
        C = load_spectra(int(r.mouse), int(r.day))[0]
        v, m = C.dv[C.region == 'VIS'], C.dv[C.region == 'MEC']
        att.append((v.max() + m.min()) / 2 if len(v) and len(m) else np.nan)
    R['atlas_border'] = att
    R.to_csv(f'{PS}/rule_boundaries.csv', index=False)
    S = R.groupby(['mouse', 'shank']).agg(boundary=('boundary', 'mean'), exit=('exit', 'mean'),
                                          top=('top', 'mean'), n_sessions=('day', 'size'),
                                          boundary_sd=('boundary', 'std')).reset_index()
    S.to_csv(f'{PS}/rule_shanks.csv', index=False)
    Sd = S.set_index(['mouse', 'shank'])
    rows = []
    for _, c in CUR.iterrows():
        mo = int(c.mouse); k = shank_of(mo, c.coord_SCs_x)
        if (mo, k) not in Sd.index:
            continue
        s = Sd.loc[(mo, k)]; y = c.coord_SCs_y
        lab = 'VIS' if y < s.boundary else 'MEC/PARA' if y <= s.exit + 30 else 'below cortex'
        rows.append(dict(mouse=mo, day=int(c.day), cluster_id=int(c.cluster_id), shank=k, dv=y,
                         rule=lab, atlas=atlas_region(c.brain_region), brain_region=c.brain_region))
    L = pd.DataFrame(rows)
    L.to_csv(f'{PS}/rule_labels.csv', index=False)
    print(R.round(0).to_string(index=False))
    print(pd.crosstab(L.atlas, L.rule))
    m = L.atlas.isin(['VIS', 'MEC/PARA'])
    print(f'agreement where the atlas says VIS or MEC/PARA: {(L.rule[m] == L.atlas[m]).mean():.1%} of {m.sum()} cells')
    print('by mouse:', L[m].groupby('mouse').apply(lambda g: round((g.rule == g.atlas).mean(), 3)).to_dict())

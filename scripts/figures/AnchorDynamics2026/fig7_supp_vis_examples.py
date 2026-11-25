"""Figure 7: entorhinal and visual anchoring share one state, in four sessions.

An alternative to Figure 7A-B: four sessions, two per row, each drawn the way
Figure 4 draws its examples.

RASTER. One population raster split into two blocks, MEC (ENTm) and visual
cortex, the way Figure 4 splits its raster by shank, each block under a grey
strip giving every cell's region class in Figure 4's grey key (MEC superficial
and deep apart). Within each block cells run from strongest to weakest agreement
with that structure's own PC1 (each cell left out of the axis), then, past a gap,
the locked cells bundled (always anchored, then always non-anchored); each block's title gives how
many of its cells are locked into one mode.

PC1. The session's anchoring axis computed SEPARATELY from each structure's own
cells (PC1 of that block's label matrix, signed to the session's anchored
fraction, z-scored), drawn as one line per structure on a shared axis, so
whether visual cortex follows the entorhinal switches is read off two lines.
Dashed lines are the session's transitions, as in Figure 7.

PROBE. The 4-shank Neuropixels 2.0 probe (probeinterface NP2014) with the
session's 384 recorded contacts dark, and every MEC and visual cluster at its
own probe position, coloured by agreement above chance (r with the session PC1
minus the mean of its circular-shift null) on the grey-white-orange scale.
Locked cells have no agreement and are drawn at chance. Beside it, the shank's
LFP spectra on the same depth axis with the Figure S15 landmarks, and the DiI
section of that shank.

E-G. IS THE STATE SHARED? Every session with at least MIN_AXIS varying cells
in each structure. Each structure's state on each trial is its own PC1 (as in
the middle panels); its switches are that PC1's sign changes with epochs
shorter than three trials absorbed (major_transitions, as the dashed lines).
(E) Switch-triggered averages: each structure's fraction of varying cells in
the anchored mode around the OTHER structure's switches and its own, signed so
the new state is up, averaged per session then across sessions, against
circular time shifts of the non-triggering structure. (F) r(MEC PC1, VIS PC1)
against the ceiling set by each axis's own reliability (split-half, Spearman-
Brown corrected): points on the diagonal share as much as their noise allows.
(G) Census: every cell as following an axis (r above the 95th percentile of
its own circular-shift null), opposing it (below the 5th), neither, or locked.
Cached in data/population_state/s15_shared_*.csv.

Sessions: the two of Figure 7 (M21 D19, M27 D23) and the two others with at
least 25 cells in each structure and the clearest contrast between them, from
mice not already shown (M26 D18, M28 D19).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

import anatomy_slice as ANA
import fig2_supp_vis_anchored as F
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         load_session_labels, major_transitions)

OUT = f'{F.FIG}/fig7_supp_vis_examples.pdf'
# Example sessions reach far into both structures (90th-percentile distance from
# the border >= MIN_EXTENT on each side), on reliable axes; M27 D23, the earlier
# example, has every visual cell within ~190 um of MEC and is now excluded.
SESSIONS = [(26, 18), (21, 19)]
# E: one session's cells that follow the MEC state, never switch, or switch on
# their own (cluster ids; structure, kind)
MAP_SESSION = (26, 18)
# depth slots (0-based) chosen for agreement rather than activity
BY_AGREE = {}
# cells forced into the examples: in M26 D18 cell 202 (the first example cell of
# Figure 1 and the graphical abstract) and 175, both clear MEC followers, and
# visual cells 230, the session's clearest visual follower that fires readably
# (7.8 Hz), 262 and 268, the most active visual cell above 262 (the superficial
# visual cells all fire below 1 Hz on the track)
FORCE = {(26, 18): {0: 202, 1: 175, 2: 230, 3: 262, 4: 268}}
MIN_SEP_UM = 150      # example cells at least this far apart in depth
MAP_CELLS = [(175, 'VIS', 'follows MEC state'), (192, 'VIS', 'locked anchored'),
             (178, 'VIS', 'switches independently'), (154, 'MEC', 'follows MEC state'),
             (84, 'MEC', 'locked anchored'), (44, 'MEC', 'switches independently')]
# Figure 4's grey key for the region strip (fig3_anatomy.REGION_GREY): MEC
# superficial, MEC deep, visual cortex
GREY = {'MEC sup': .80, 'MEC deep': .69, 'VIS': .91}
# PC1 line colours: the visual green of Figure 7, and near-black for MEC --
# Figure 7's MEC purple would read as the anchored colour on a PC1 axis
LCOL = {'MEC': '0.15', 'VIS': F.RCOL['VIS']}
TAC = ListedColormap([NONANCH_COLOR, ANCH_COLOR]); TAC.set_bad('white')
from matplotlib.colors import LinearSegmentedColormap as _LSC
AGREE_CMAP = _LSC.from_list('agree', ['#3a3a3a', '#9a9a9a', '#ffffff', '#f4a259', '#b35806'])
REG = pd.read_csv(f'{F.PS}/pc1_by_region.csv')
_DV = {(int(a), int(b), int(c)): float(y) for a, b, c, y in
       F.CELLXYZ[['mouse', 'day', 'cluster_id', 'coord_SCs_y']].itertuples(index=False)}
MIN_AXIS = 10            # varying cells needed to build a structure's PC1
# H: agreement against distance from the border, single-shank sessions only
BORDER_CSV = f'{F.PS}/s15_border_agreement.csv'
SESS_CSV = f'{F.PS}/s15_border_sessions.csv'
MIN_CELLS = 8            # scored cells for a session's within-session rho
MIN_HALF = 5             # varying visual cells in each half of the near/far split
MAX_SPAN = 100.          # um, mediolateral span of a single-shank session
N_SHIFT = 100
BORDER_AXIS_EXCLUDE_UM = 0.0   # I: axes from all cells
WIN = 15                 # trials either side of a switch (E)
N_TRIG_NULL = 200        # time shifts per session for E's null
SH_SESS = f'{F.PS}/s15_shared_sessions.csv'
SH_CELL = f'{F.PS}/s15_shared_cells.csv'
SH_TRIG = f'{F.PS}/s15_shared_triggered.npz'


def _cls(acr):
    a = str(acr)
    if a.startswith('ENTm'):
        return 'MEC sup' if a[4:5] in ('1', '2', '3') else 'MEC deep'
    return 'VIS' if a.startswith('VIS') else None


def _z(v):
    return (v - np.nanmean(v)) / np.nanstd(v)


def blocks(mo, dy):
    z = load_session_labels(mo, dy)
    ids = np.array([int(c) for c in z['cluster_id']])
    lab = np.asarray(z['labels'], float)
    frac = np.asarray(z['frac_anch'], float)
    rr = REG[(REG.mouse == mo) & (REG.day == dy)].set_index('cluster_id')
    out = {}
    for name in ('MEC', 'VIS'):
        rows = [i for i, c in enumerate(ids) if c in rr.index
                and (_cls(rr.brain_region[c]) or '').startswith(name)
                and not np.isnan(lab[i]).all()]
        L = lab[rows]; cid = ids[rows]
        cls = np.array([_cls(rr.brain_region[c]) for c in cid])
        varies = np.nanstd(L, axis=1) > 0
        # this structure's own PC1, signed to the session's anchored fraction
        Mc = np.nan_to_num(L[varies] - np.nanmean(L[varies], axis=1, keepdims=True))
        pc = np.linalg.svd(Mc, full_matrices=False)[2][0]
        if np.corrcoef(pc, np.nan_to_num(frac))[0, 1] < 0:
            pc = -pc
        # cells ordered by agreement with their own structure's PC1 (signed r,
        # each cell left out of the axis it is scored against), strongest first;
        # locked cells, which have no agreement, at the right end
        agree = np.full(len(L), -np.inf)
        vi = np.flatnonzero(varies)
        for k, i in enumerate(vi):
            others = np.delete(vi, k)
            Mo = np.nan_to_num(L[others] - np.nanmean(L[others], axis=1, keepdims=True))
            p_ = np.linalg.svd(Mo, full_matrices=False)[2][0]
            if np.corrcoef(p_, pc)[0, 1] < 0:
                p_ = -p_
            agree[i] = np.corrcoef(np.nan_to_num(L[i] - np.nanmean(L[i])), p_)[0, 1]
        # switching cells by agreement, then a gap, then the locked cells bundled:
        # always-anchored together, then always-non-anchored
        ov = vi[np.argsort(-agree[vi], kind='stable')]
        lk = np.flatnonzero(~varies)
        ol = lk[np.argsort(-np.nanmean(L[lk], axis=1), kind='stable')]
        gap = max(2, int(round(.03 * len(L)))) if len(ol) else 0
        M = np.vstack([L[ov], np.full((gap, L.shape[1]), np.nan), L[ol]])
        strip = np.r_[[GREY[c] for c in cls[ov]], [1.] * gap, [GREY[c] for c in cls[ol]]]
        out[name] = dict(agree={int(cid[i]): float(agree[i]) for i in vi},
                         M=M, strip=strip, n_var=int(varies.sum()), n_lock=int((~varies).sum()),
                         pc=_z(pc), n=len(L))
    return out, major_transitions(np.nan_to_num(np.asarray(z['pc1'], float))), rr


def _region_axis(L, frac):
    """PC1 of a structure's varying cells, signed to the anchored fraction."""
    v = np.nanstd(L, axis=1) > 0
    Mc = np.nan_to_num(L[v] - np.nanmean(L[v], axis=1, keepdims=True))
    pc = np.linalg.svd(Mc, full_matrices=False)[2][0]
    return -pc if np.corrcoef(pc, np.nan_to_num(frac))[0, 1] < 0 else pc


def _shift_r(x, y, guard=5):
    """r(x, y) and r for every circular shift of x by >= guard trials (FFT)."""
    x = (x - x.mean()) / (x.std() + 1e-12); y = (y - y.mean()) / (y.std() + 1e-12)
    n = len(x)
    c = np.real(np.fft.ifft(np.conj(np.fft.fft(x)) * np.fft.fft(y))) / n
    sh = np.r_[guard:n - guard + 1]
    return float(c[0]), c[sh]


def _switches(pc):
    """Switch trials and direction (+1 into anchored) of a PC1."""
    t = major_transitions(pc)
    d = [np.sign(np.mean(pc[k:k + 3])) or 1. for k in t]
    return np.array(t, int), np.array(d)


def _split_half(L, frac, rng, n=100):
    """Spearman-Brown reliability of a structure's PC1 from random half splits."""
    rs = []
    for _ in range(n):
        p = rng.permutation(len(L)); h = len(L) // 2
        rs.append(abs(np.corrcoef(_region_axis(L[p[:h]], frac),
                                  _region_axis(L[p[h:]], frac))[0, 1]))
    r = float(np.mean(rs))
    return 2 * r / (1 + r)


def shared_state(exclude=False):
    """Per session (F, and the traces behind E) and per cell (G)."""
    sfx = f'_ex{int(EXCLUDE_UM)}' if exclude else ''
    f_s, f_c, f_t = (SH_SESS.replace('.csv', sfx + '.csv'), SH_CELL.replace('.csv', sfx + '.csv'),
                     SH_TRIG.replace('.npz', sfx + '.npz'))
    if all(os.path.exists(f) for f in (f_s, f_c, f_t)):
        return pd.read_csv(f_s), pd.read_csv(f_c), dict(np.load(f_t))
    near = near_border() if exclude else set()
    srows, crows, trig = [], [], {}
    for (mo, dy), g in REG.groupby(['mouse', 'day']):
        z = load_session_labels(int(mo), int(dy))
        if z is None:
            continue
        ids = np.array([int(c) for c in z['cluster_id']])
        lab = np.asarray(z['labels'], float)
        frac = np.asarray(z['frac_anch'], float)
        g = g.set_index('cluster_id')
        struct = np.array([('' if (int(mo), int(dy), int(c)) in near else
                            'MEC' if str(g.brain_region.get(c, '')).startswith('ENTm') else
                            'VIS' if str(g.brain_region.get(c, '')).startswith('VIS') else '')
                           for c in ids])
        ok = ~np.isnan(lab).all(axis=1)
        var = ok & (np.nanstd(lab, axis=1) > 0)
        if min((var & (struct == k)).sum() for k in ('MEC', 'VIS')) < MIN_AXIS:
            continue
        rng = np.random.default_rng(int(mo) * 100 + int(dy))
        ax_ = {k: _region_axis(lab[ok & (struct == k)], frac) for k in ('MEC', 'VIS')}
        # fraction of each structure's varying cells in the anchored mode
        fr = {k: np.nanmean(lab[var & (struct == k)], axis=0) for k in ('MEC', 'VIS')}
        n_tr = lab.shape[1]
        # ---- E: switch-triggered averages, per session, both triggers
        for trg in ('MEC', 'VIS'):
            t, d = _switches(ax_[trg])
            t_ok = (t >= WIN) & (t <= n_tr - WIN)
            t, d = t[t_ok], d[t_ok]
            if not len(t):
                continue
            lags = np.arange(-WIN, WIN)
            for k in ('MEC', 'VIS'):
                X = np.array([dd * (fr[k][tt + lags] - fr[k][tt - 5:tt].mean())
                              for tt, dd in zip(t, d)])
                trig.setdefault(f'{trg}>{k}', []).append(X.mean(0))
                # null: the same switches, the traced structure shifted in time
                N = []
                for s_ in rng.integers(WIN, n_tr - WIN, N_TRIG_NULL):
                    f_ = np.roll(fr[k], s_)
                    N.append(np.mean([dd * (f_[tt + lags] - f_[tt - 5:tt].mean())
                                      for tt, dd in zip(t, d)], axis=0))
                trig.setdefault(f'{trg}>{k}_null', []).append(np.array(N))
            trig.setdefault(f'{trg}_n', []).append(len(t))
        # ---- F: r against the reliability ceiling
        r_mv, null = _shift_r(ax_['MEC'], ax_['VIS'], guard=10)
        relM = _split_half(lab[ok & (struct == 'MEC')], frac, rng)
        relV = _split_half(lab[ok & (struct == 'VIS')], frac, rng)
        srows.append(dict(mouse=int(mo), day=int(dy), n_trials=n_tr,
                          n_mec=int((var & (struct == 'MEC')).sum()),
                          n_vis=int((var & (struct == 'VIS')).sum()),
                          r_mv=r_mv, null95=float(np.percentile(np.abs(null), 95)),
                          p_mv=float((np.sum(np.abs(null) >= abs(r_mv)) + 1) / (len(null) + 1)),
                          rel_mec=relM, rel_vis=relV,
                          ceiling=float(np.sqrt(max(relM, 0) * max(relV, 0)))))
        # ---- G: every cell against each structure's axis (own: leave-one-out)
        for i in np.flatnonzero(ok & np.isin(struct, ['MEC', 'VIS'])):
            own = struct[i]
            row = dict(mouse=int(mo), day=int(dy), cluster_id=int(ids[i]), structure=own,
                       locked=bool(not var[i]))
            if var[i]:
                for k in ('MEC', 'VIS'):
                    if k == own:
                        m_ = ok & (struct == k); m_[i] = False
                        pc = _region_axis(lab[m_], frac)
                    else:
                        pc = ax_[k]
                    r, nl = _shift_r(np.nan_to_num(lab[i]), pc)
                    row[f'r_{k}'] = r
                    row[f'cat_{k}'] = ('follows' if r > np.percentile(nl, 95) else
                                       'opposes' if r < np.percentile(nl, 5) else 'neither')
            crows.append(row)
        print(f'  M{mo}D{dy}: r {r_mv:+.2f}, ceiling {srows[-1]["ceiling"]:.2f}', flush=True)
    S, C = pd.DataFrame(srows), pd.DataFrame(crows)
    T = {}
    for k, v in trig.items():
        T[k] = np.array(v) if not k.endswith('_null') else np.stack(v)   # sessions x ...
    for X, f in ((S, f_s), (C, f_c)):
        X.to_csv(f + '.partial', index=False); os.replace(f + '.partial', f)
    np.savez(f_t + '.partial.npz', **T); os.replace(f_t + '.partial.npz', f_t)
    return S, C, T


def _excess(x, pc, rng):
    r = float(np.corrcoef(x, pc)[0, 1])
    null = np.abs([np.corrcoef(np.roll(x, s_), pc)[0, 1]
                   for s_ in rng.integers(5, len(x) - 5, N_SHIFT)])
    return r, abs(r) - float(np.nanmean(null))


def _xcorr_lag(a, b, max_lag=10):
    """Lag (trials) of the peak cross-correlation; positive = b lags a."""
    best, lag = -np.inf, 0
    for l in range(-max_lag, max_lag + 1):
        x, y = (a[:len(a) - l], b[l:]) if l >= 0 else (a[-l:], b[:len(b) + l])
        c = np.corrcoef(x, y)[0, 1]
        if c > best:
            best, lag = c, l
    return lag


def single_shank_sessions():
    """Sessions whose cells all sit on one shank: mediolateral span < MAX_SPAN.

    Most devices give each cell its shank's one nominal position; mouse 21's
    give a continuous per-cell estimate, spanning < 40 um within a shank. Shank
    pitch is 210-270 um, so MAX_SPAN separates the two cleanly.
    """
    c = F.CELLXYZ.assign(ml=F.CELLXYZ.coord_SCs_x.abs())
    span = c.groupby(['mouse', 'day']).ml.agg(lambda v: v.max() - v.min())
    return set(span[span < MAX_SPAN].index)


def border_agreement():
    """Per cell and per session, in the single-shank sessions.

    Cells: distance to the other structure, agreement above chance with the
    other structure's PC1 (E, F), and, for every cell, with its OWN
    structure's PC1 rebuilt without it (H).
    Sessions: the two PC1s compared (G) and the visual PC1 rebuilt from the
    visual cells nearer and farther than the session's median distance to MEC (I).
    """
    if os.path.exists(BORDER_CSV) and os.path.exists(SESS_CSV):
        return pd.read_csv(BORDER_CSV), pd.read_csv(SESS_CSV)
    from scipy.spatial import cKDTree
    ann, id2 = ANA._atlas()
    want = {k: {i for i, a in id2.items() if str(a).startswith(p)}
            for k, p in (('MEC', 'ENTm'), ('VIS', 'VIS'))}
    cx = F.CELLXYZ.set_index(['mouse', 'day', 'cluster_id'])
    single = single_shank_sessions()
    rows, srows = [], []
    for (mo, dy), g in REG.groupby(['mouse', 'day']):
        if (mo, dy) not in single:
            continue
        z = load_session_labels(int(mo), int(dy))
        if z is None:
            continue
        ids = np.array([int(c) for c in z['cluster_id']])
        lab = np.asarray(z['labels'], float)
        frac = np.asarray(z['frac_anch'], float)
        g = g.set_index('cluster_id')
        struct = np.array([('MEC' if str(g.brain_region.get(c, '')).startswith('ENTm') else
                            'VIS' if str(g.brain_region.get(c, '')).startswith('VIS') else '')
                           for c in ids])
        ok = ~np.isnan(lab).all(axis=1)
        var = ok & (np.nanstd(lab, axis=1) > 0)
        if min((var & (struct == k)).sum() for k in ('MEC', 'VIS')) < MIN_AXIS:
            continue
        axes = {k: _region_axis(lab[ok & (struct == k)], frac) for k in ('MEC', 'VIS')}
        key = [(int(mo), int(dy), int(c)) for c in ids]
        have = np.array([k in cx.index for k in key])
        P = np.full((len(ids), 3), np.nan)
        P[have] = ANA.to_ccf(cx.loc[[k for k, h in zip(key, have) if h]].reset_index(),
                             ('coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x'))
        lo = np.maximum((np.nanmin(P, 0) / 10).astype(int) - 150, 0)
        hi = np.minimum((np.nanmax(P, 0) / 10).astype(int) + 150, ann.shape)
        cube = np.asarray(ann[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        trees = {}
        for k in ('MEC', 'VIS'):
            vox = np.argwhere(np.isin(cube, list(want[k])))
            trees[k] = cKDTree((vox + lo) * 10.) if len(vox) else None
        if trees['MEC'] is None or trees['VIS'] is None:
            continue
        # I uses every cell for both structures' axes (BORDER_AXIS_EXCLUDE_UM = 0);
        # set it to EXCLUDE_UM to build them, as F-H do, without near-border cells.
        dcell = np.full(len(ids), np.nan)
        for k, other_ in (('MEC', 'VIS'), ('VIS', 'MEC')):
            mk = have & (struct == k)
            if mk.any():
                dcell[mk] = trees[other_].query(P[mk])[0]
        axis_ok = ok & (np.nan_to_num(dcell, nan=np.inf) >= BORDER_AXIS_EXCLUDE_UM)
        if min((axis_ok & var & (struct == k)).sum() for k in ('MEC', 'VIS')) < MIN_AXIS:
            continue
        axes = {k: _region_axis(lab[axis_ok & (struct == k)], frac) for k in ('MEC', 'VIS')}
        rng = np.random.default_rng(int(mo) * 100 + int(dy))
        for i in np.flatnonzero(var & have & np.isin(struct, ['MEC', 'VIS'])):
            own = struct[i]; other = 'VIS' if own == 'MEC' else 'MEC'
            x = np.nan_to_num(lab[i])
            r, ex = _excess(x, axes[other], rng)
            # own axis without the cell itself
            mine = axis_ok & (struct == own); mine[i] = False
            r_own, ex_own = _excess(x, _region_axis(lab[mine], frac), rng)
            rows.append(dict(mouse=int(mo), day=int(dy), cluster_id=int(ids[i]),
                             structure=own, other=other,
                             dist=float(trees[other].query(P[i])[0]),
                             r=r, excess=ex, r_own=r_own, excess_own=ex_own))
        # ---- the two axes against each other
        m, v = axes['MEC'], axes['VIS']
        r_mv = float(np.corrcoef(m, v)[0, 1])
        null = np.array([np.corrcoef(m, np.roll(v, s_))[0, 1]
                         for s_ in rng.integers(10, len(v) - 10, 1000)])
        same = float(np.mean(np.sign(_z(m)) == np.sign(_z(v))))
        # ---- the visual axis from near and far visual cells
        iv = np.flatnonzero(var & have & (struct == 'VIS'))
        dv = np.array([trees['MEC'].query(P[i])[0] for i in iv])
        near, far = iv[dv <= np.median(dv)], iv[dv > np.median(dv)]
        r_near = r_far = np.nan
        if len(near) >= MIN_HALF and len(far) >= MIN_HALF:
            r_near = float(np.corrcoef(m, _region_axis(lab[near], frac))[0, 1])
            r_far = float(np.corrcoef(m, _region_axis(lab[far], frac))[0, 1])
        srows.append(dict(mouse=int(mo), day=int(dy), n_trials=len(m),
                          n_mec=int((var & (struct == 'MEC')).sum()),
                          n_vis=int((var & (struct == 'VIS')).sum()),
                          r_mv=r_mv, null95=float(np.percentile(np.abs(null), 95)),
                          p_mv=float((np.sum(np.abs(null) >= abs(r_mv)) + 1) / (len(null) + 1)),
                          sign_agree=same, lag=_xcorr_lag(_z(m), _z(v)),
                          d_split=float(np.median(dv)), d_near=float(np.median(dv[dv <= np.median(dv)])),
                          d_far=float(np.median(dv[dv > np.median(dv)])),
                          r_near=r_near, r_far=r_far))
        print(f'  M{mo}D{dy}: r(MEC,VIS) {r_mv:+.2f}, near {r_near:+.2f}, far {r_far:+.2f}', flush=True)
    D, S = pd.DataFrame(rows), pd.DataFrame(srows)
    for X, f in ((D, BORDER_CSV), (S, SESS_CSV)):
        X.to_csv(f + '.partial', index=False); os.replace(f + '.partial', f)
    return D, S


def border_stats(d, y='excess'):
    from scipy.stats import spearmanr, wilcoxon
    import statsmodels.formula.api as smf
    d = d.rename(columns={y: '_y'})
    ws = [spearmanr(g.dist, g._y)[0] for _, g in d.groupby(['mouse', 'day'])
          if len(g) >= MIN_CELLS and g.dist.nunique() > 2]
    ws = np.array([w for w in ws if np.isfinite(w)])
    dd = d.assign(sess=d.mouse.astype(str) + '_' + d.day.astype(str), dist_mm=d.dist / 1000)
    fit = smf.mixedlm('_y ~ dist_mm', dd, groups=dd['sess']).fit(method='lbfgs')
    return dict(n_cells=len(d), n_sess=d.groupby(['mouse', 'day']).ngroups,
                rho_pooled=spearmanr(d.dist, d._y)[0], n_ws=len(ws),
                rho_ws=float(np.median(ws)), p_ws=float(wilcoxon(ws).pvalue),
                neg_ws=int((ws < 0).sum()), b=float(fit.params['dist_mm']),
                p_lmm=float(fit.pvalues['dist_mm']))


EXTENT_CSV = f'{F.PS}/s15_extent.csv'
CELL_DIST_CSV = f'{F.PS}/s15_cell_dist.csv'
EXCLUDE_UM = 100.0      # cells nearer than this to the other structure are left out of F-I
MIN_EXTENT = 500.0       # um: 90th-percentile distance from the border, each side


def session_extent():
    """How far each session's cells reach from the VIS-MEC border, on each side.

    Every MEC and visual cell's 3D atlas distance to the nearest voxel of the
    other structure; a session's extent on a side is the 90th percentile.
    Sessions whose cells all sit near the border are where an assignment
    error between the two structures would matter most, and they cannot show
    whether agreement changes with distance.
    """
    if os.path.exists(EXTENT_CSV) and os.path.exists(CELL_DIST_CSV):
        return pd.read_csv(EXTENT_CSV).set_index(['mouse', 'day'])
    from scipy.spatial import cKDTree
    ann, id2 = ANA._atlas()
    want = {k: {i for i, a in id2.items() if str(a).startswith(p)}
            for k, p in (('MEC', 'ENTm'), ('VIS', 'VIS'))}
    rows, cdist = [], []
    for (mo, dy), g in F.CELLXYZ[F.CELLXYZ.macro.isin(['MEC', 'VIS'])].groupby(['mouse', 'day']):
        if set(g.macro) != {'MEC', 'VIS'}:
            continue
        P = ANA.to_ccf(g, ('coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x'))
        lo = np.maximum((P.min(0) / 10).astype(int) - 150, 0)
        hi = np.minimum((P.max(0) / 10).astype(int) + 150, ann.shape)
        cube = np.asarray(ann[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        r = dict(mouse=int(mo), day=int(dy))
        for st, other in (('VIS', 'MEC'), ('MEC', 'VIS')):
            vox = np.argwhere(np.isin(cube, list(want[other])))
            m = (g.macro == st).values
            if not len(vox):
                r[f'p90_{st}'] = np.nan; continue
            d = cKDTree((vox + lo) * 10.).query(P[m])[0]
            r[f'p90_{st}'] = float(np.percentile(d, 90)); r[f'n_{st}'] = int(m.sum())
            cdist.append(pd.DataFrame(dict(mouse=int(mo), day=int(dy),
                                           cluster_id=g.cluster_id.values[m].astype(int),
                                           structure=st, dist=d)))
        rows.append(r)
    E = pd.DataFrame(rows)
    E['extent'] = E[['p90_VIS', 'p90_MEC']].min(1)
    E.to_csv(EXTENT_CSV, index=False)
    pd.concat(cdist, ignore_index=True).to_csv(CELL_DIST_CSV, index=False)
    return E.set_index(['mouse', 'day'])


def near_border():
    """(mouse, day, cluster_id) of cells within EXCLUDE_UM of the other structure."""
    session_extent()
    c = pd.read_csv(CELL_DIST_CSV)
    c = c[c.dist < EXCLUDE_UM]
    return set(zip(c.mouse.astype(int), c.day.astype(int), c.cluster_id.astype(int)))


PAIR_MIN_SHIFT = 10      # trials: the circular-shift null for pairwise agreement


def pair_matrix(mo, dy):
    """Cell x cell agreement of anchoring labels above a circular-shift null.

    Cells in the order of the rasters: visual then MEC, each dorsal -> ventral.
    For each pair, the correlation of their label sequences against that pair
    with one sequence circularly shifted by every offset of at least
    PAIR_MIN_SHIFT trials (FFT): r minus the null's signed mean. Cells that never
    switch have no sequence to correlate and are left out.
    """
    z = load_session_labels(mo, dy)
    ids = np.array([int(c) for c in z['cluster_id']])
    lab = np.asarray(z['labels'], float)
    rr = REG[(REG.mouse == mo) & (REG.day == dy)].set_index('cluster_id')
    order, n_vis = [], 0
    for name in ('VIS', 'MEC'):
        rows = [i for i, c in enumerate(ids) if c in rr.index
                and (_cls(rr.brain_region[c]) or '').startswith(name) and not np.isnan(lab[i]).all()]
        dv = np.array([_DV.get((mo, dy, int(ids[i])), np.inf) for i in rows])
        rows = [rows[k] for k in np.argsort(dv, kind='stable')]
        order += rows
        if name == 'VIS':
            n_vis = len(rows)
    L = lab[order]
    L = np.where(np.isnan(L), np.nanmean(L, axis=1, keepdims=True), L)
    var = L.std(1) > 0
    Zs = np.zeros_like(L)
    Zs[var] = (L[var] - L[var].mean(1, keepdims=True)) / L[var].std(1, keepdims=True)
    T = L.shape[1]
    Fz = np.fft.fft(Zs, axis=1)
    sh = np.r_[PAIR_MIN_SHIFT:T - PAIR_MIN_SHIFT + 1]
    Zm = np.full((len(L), len(L)), np.nan)
    for i in np.flatnonzero(var):
        xc = np.real(np.fft.ifft(np.conj(Fz[i])[None] * Fz, axis=1)) / T      # [cells, lags]
        nul = xc[:, sh]
        # agreement above chance on the correlation's own scale: r minus the
        # SIGNED mean of the shifted r (about 0). A z-score divides by the
        # null's spread, which a session with one or two long state blocks
        # inflates, and so hides strong agreement exactly where it is clearest.
        Zm[i] = xc[:, 0] - nul.mean(1)
    np.fill_diagonal(Zm, np.nan)
    # locked cells have no sequence to correlate; they are dropped (their
    # counts are in the raster headers) rather than drawn as grey stripes
    n_vis = int(var[:n_vis].sum())
    return Zm[np.ix_(var, var)], n_vis



def pick_followers(mo, dy, n=5, by_agree=(), force=None, min_vis=2, dorsal=False):
    """n switching (not locked) cells, at least min_vis of them visual.

    Forced cells are taken first; with dorsal, then (unless forced cells fill
    the visual slots) the most active of the three most dorsal switching
    visual cells. Visual slots are then filled at evenly
    spaced depths through the switching visual cells, and the remaining slots
    at evenly spaced depths through the switching MEC cells, never within
    MIN_SEP_UM of a cell already taken; at each target, of the three nearest
    candidates, the one with the highest mean rate on the track (so its map
    is readable). Agreement is not used to choose. Numbered dorsal to ventral.
    """
    B, _, rr = blocks(mo, dy)
    # mean rate on the track, from the rate maps the panel shows (the unit
    # table's firing_rate is missing for a third of cells)
    ids = [c for name in ('VIS', 'MEC') for c in B[name]['agree']]
    mp = F.example_maps(mo, dy, ids)
    cand = []
    for name in ('VIS', 'MEC'):
        for c, a in B[name]['agree'].items():
            d = _DV.get((mo, dy, c), np.nan)
            if np.isfinite(d):
                r = np.nanmean(mp[c]) if c in mp else np.nan
                cand.append((c, name, a, d, float(r) if np.isfinite(r) else 0.))
    byid = {x[0]: x for x in cand}
    out = [byid[c] for c in (force or {}).values()]

    def fill(name, k):
        pool = [x for x in cand if x[1] == name]
        if not pool or k <= 0:
            return
        d = np.array([x[3] for x in pool])
        for tgt in np.linspace(d.min(), d.max(), k + 2)[1:-1] if k < len(pool) else d:
            free = [x for x in pool if all(x[0] != y[0] and abs(x[3] - y[3]) >= MIN_SEP_UM for y in out)]
            if not free:
                free = [x for x in pool if all(x[0] != y[0] for y in out)]
            if not free:
                return
            near = sorted(free, key=lambda x: abs(x[3] - tgt))[:3]
            out.append(max(near, key=lambda x: x[4]))

    if dorsal and sum(x[1] == 'VIS' for x in out) < min_vis:
        top = sorted((x for x in cand if x[1] == 'VIS' and x not in out), key=lambda x: x[3])[:3]
        if top:
            out.append(max(top, key=lambda x: x[4]))
    fill('VIS', min_vis - sum(x[1] == 'VIS' for x in out))
    fill('MEC', n - len(out))
    return sorted((x[:4] for x in out), key=lambda x: x[3])

def maps_panel(fig, gs, mo, dy, picks, letter):
    """Trial x position rate maps of the picked cells, each with its own label strip."""
    from spatial_manifolds.anchoring import load_session_labels as _lsl
    z = _lsl(mo, dy)
    ids_ = [int(c) for c in z['cluster_id']]
    tr = major_transitions(np.nan_to_num(np.asarray(z['pc1'], float)))
    mp = F.example_maps(mo, dy, [c for c, *_ in picks])
    gm = gs.subgridspec(1, 2 * len(picks), width_ratios=[1, .09] * len(picks), wspace=.08)
    dv0 = min(d for *_, d in picks)
    ax0 = None
    for k, (c, st, a, dv) in enumerate(picks):
        ax = fig.add_subplot(gm[2 * k], **({} if ax0 is None else {'sharey': ax0}))
        ax0 = ax0 or ax
        M = mp[c]
        ax.imshow(np.clip(M / (np.nanpercentile(M, 99) or 1), 0, 1), aspect='auto', cmap=F.CMAP,
                  vmin=0, vmax=1, interpolation='nearest', extent=[0, 200, M.shape[0], 0])
        for t_ in tr:
            ax.axhline(t_, color='w', lw=.5, ls='--', alpha=.7)
        ax.set_title(f'{k + 1}  {st} cell {c}\nr {a:+.2f}, +{dv - dv0:.0f} um', fontsize=5.6,
                     color=LCOL[st] if st == 'VIS' else '0.15', pad=2, linespacing=1.1, loc='left')
        ax.set_xticks([0, 200]); ax.tick_params(labelsize=5.6)
        ax.set_xlabel('cm', fontsize=6, labelpad=0)
        if k == 0:
            ax.set_ylabel('Trial', fontsize=7.5)
            ax.text(-.4, 1.12, letter, transform=ax.transAxes, fontsize=10, weight='bold')
        else:
            ax.tick_params(labelleft=False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        # the cell's own anchoring label on every trial
        axl = fig.add_subplot(gm[2 * k + 1], sharey=ax0)
        lab = np.asarray(z['labels'][ids_.index(c)], float)
        axl.imshow(np.ma.masked_invalid(lab)[:, None], aspect='auto', cmap=TAC, vmin=0, vmax=1,
                   interpolation='nearest', extent=[0, 1, len(lab), 0])
        axl.set_xticks([]); axl.tick_params(labelleft=False, left=False)
        for sp in axl.spines.values():
            sp.set_visible(False)



THETA_MOD = 0.07     # open-field theta index above which a cell counts as theta-modulated
THETA_WIN_UM = 200.0  # depth window for the running fraction


def all_theta_index(mo, dy):
    """Theta index for EVERY cluster of a session, pooled over OF1, OF2 and VR.

    The open-field index (of_theta_index.py) needs >= 200 spikes in one open
    field, which drops quiet cells, and it was computed on curated clusters
    only. Here each cluster's rate spectrum (the same band contrast: 6-10 Hz
    against 3-5 and 11-13 Hz, 2 ms bins, 5 ms smoothing, Welch) is computed
    in each of its three sessions that has >= MIN_SPK_SESSION spikes, the
    spectra averaged weighted by recording duration, and the index taken from
    the average, so every cluster with enough spikes overall is scored.
    """
    import os as _os
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import welch
    out = f'{F.PS}/theta_index_all_M{mo}D{dy}.csv'
    if _os.path.exists(out):
        return pd.read_csv(out).set_index('cluster_id').theta_index
    import pynapple as _nap
    src = '/Volumes/datasetsSD/Ephys_data/clark2025'
    acc = {}
    for ses in ('OF1', 'OF2', 'VR'):
        p = f'{src}/M{mo}/D{dy}/{ses}/sub-M{mo}_ses-D{dy}_typ-{ses}_srt-kilosort4_clusters.npz'
        if not _os.path.exists(p):
            continue
        cl = _nap.load_file(p)
        for c in [int(x) for x in cl.index]:
            t = np.asarray(cl[c].index, float)
            if len(t) < MIN_SPK_SESSION or t.max() - t.min() < 120:
                continue
            edges = np.arange(t.min(), t.max() + .002, .002)
            r = gaussian_filter1d(np.histogram(t, bins=edges)[0] / .002, 2.5)
            f, P = welch(r, fs=500., nperseg=min(2048, len(r)), noverlap=min(2048, len(r)) // 2)
            w = t.max() - t.min()
            a = acc.setdefault(c, [np.zeros_like(P), 0., f, 0])
            if len(a[0]) == len(P):
                a[0] += P * w; a[1] += w; a[3] += len(t)
    rows = []
    for c, (P, w, f, n) in acc.items():
        P = P / w
        th = (f >= 6) & (f <= 10); fl = ((f >= 3) & (f <= 5)) | ((f >= 11) & (f <= 13))
        q, b = P[th].mean(), P[fl].mean()
        rows.append(dict(cluster_id=c, theta_index=(q - b) / (q + b) if q + b > 0 else np.nan, n_spikes=n))
    D_ = pd.DataFrame(rows)
    D_.to_csv(out, index=False)
    return D_.set_index('cluster_id').theta_index


MIN_SPK_SESSION = 20


# histology sections per mouse: (image, sections, um per px)
HIST = {26: ('M26_sections.png', 17, 500 / 85), 21: ('M21_sections.png', 17, 500 / 77)}


def hist_section(mo, shank):
    """The section holding this shank's DiI track: the shank's peak of DiI per
    section (red above the blue counterstain), peaks taken lateral -> medial
    and matched to shanks in mediolateral order, as Figure S16."""
    from PIL import Image
    from scipy.signal import find_peaks
    Image.MAX_IMAGE_PIXELS = None
    fn, n, um = HIST[mo]
    a = np.asarray(Image.open(f'{F.ROOT}/data/histology/{fn}').convert('RGB')).astype(np.float32)
    W = a.shape[1] / n
    ex = np.clip(a[..., 0] - .6 * a[..., 2] - 40, 0, None)
    sig = np.array([ex[:, int(round(i * W)):int(round((i + 1) * W))].sum() for i in range(n)]); sig /= sig.max()
    pk, _ = find_peaks(np.r_[0, sig, 0], prominence=.12); pk = sorted(pk - 1)
    if len(pk) == 3:
        rest = np.arange(pk[-1] + 1, n); pk.append(int(rest[np.argmax(sig[rest])]))
    pk = pk[:4]                                     # lateral -> medial
    sec = pk[3 - shank]
    if sig[sec] < .2:
        # no clear DiI track for this shank: better no section than a wrong one
        return None
    a, ex = (x[:, int(round(sec * W)):int(round((sec + 1) * W))] for x in (a, ex))
    # the whole section, trimmed only to the tissue (DAPI) with a 100 um margin
    from scipy.ndimage import uniform_filter
    m_ = uniform_filter(a[..., 2], 15) > 30
    r_, c_ = np.nonzero(m_) if m_.any() else ([0, a.shape[0] - 1], [0, a.shape[1] - 1])
    pad = int(100 / um)
    r0, r1 = max(0, np.min(r_) - pad), min(a.shape[0], np.max(r_) + pad)
    c0, c1 = max(0, np.min(c_) - pad), min(a.shape[1], np.max(c_) + pad)
    return a[r0:r1, c0:c1].astype(np.uint8), sec + 1, um


def lfp_panel(ax, axsl, mo, dy):
    """The shank's LFP spectra on the slice's depth axis, with the landmarks."""
    import anatomy_rules as AR
    got = AR.load_spectra(mo, dy)
    C, f, LP = got
    Z = (LP - LP.mean(0)) / LP.std(0)
    o = np.argsort(C.dv.values); dvc = C.dv.values[o]; Z = AR.clean_channels(Z[o])
    # map stereotaxic depth onto the slice's vertical axis through the cells
    c = axsl._cells; ok = np.isfinite(c['dv'])
    a_, b_ = np.polyfit(c['dv'][ok], c['pv'][ok], 1)
    to_v = lambda y: a_ * np.asarray(y, float) + b_
    pvc = to_v(dvc); step = np.median(np.diff(pvc))
    ax.imshow(Z, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5, interpolation='nearest',
              extent=[0, len(f), pvc[-1] + step / 2, pvc[0] - step / 2])
    ax.set_xticks([np.argmin(abs(f - x)) for x in (10, 60, 200)]); ax.set_xticklabels([10, 60, 200], fontsize=5.5)
    ax.set_xlabel('Hz', fontsize=6.5, labelpad=1)
    ax.tick_params(labelleft=False, left=False)
    r = AR.session_rule(mo, dy)
    vis, mec = C.dv[C.region == 'VIS'], C.dv[C.region == 'MEC']
    for y, col, ls, lw in ((r['boundary'], '#e66101', '-', 1.3), (r['exit'], '#5e3c99', '-', 1.1),
                           ((vis.max() + mec.min()) / 2 if len(vis) and len(mec) else np.nan, 'k', '--', .8)):
        if np.isfinite(y):
            for a2 in (ax, axsl):
                a2.axhline(to_v(y), color=col, ls=ls, lw=lw, zorder=6)
    ax.set_title('LFP spectra', fontsize=6.3, pad=2)
    print(f'LFP panel M{mo}D{dy}: boundary {r["boundary"]:.0f}, exit {r["exit"]:.0f} um')


def draw_probe(ax, mo, dy, rr, vlim, marks=()):
    """The 4-shank Neuropixels 2.0 probe (probeinterface NP2014), the session's
    recorded contacts dark, and every curated MEC and visual unit at its own probe
    position, coloured by agreement as the slice was."""
    import probeinterface as pi
    from probeinterface.plotting import plot_probe
    probe = pi.get_probe('imec', 'NP2014')
    A = pd.read_csv(f'/Volumes/datasetsSD/Ephys_data/clark2025/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv')
    on = set(zip(A.coord_probe_x.round().astype(int), A.coord_probe_y.round().astype(int)))
    xy = np.round(probe.contact_positions).astype(int)
    act = np.array([(x, y) in on for x, y in xy])
    plot_probe(probe, ax=ax, contacts_kargs=dict(alpha=0),
               probe_shape_kwargs=dict(facecolor='0.97', edgecolor='0.6', lw=.4, alpha=1),
               title=False)
    # the pads drawn here rather than by plot_probe, whose contacts vanish at
    # this scale: 12 um squares, the recorded 384 dark
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    for sel, col in ((~act, '0.82'), (act, '0.25')):
        ax.add_collection(PatchCollection([Rectangle((x - 6, y - 6), 12, 12) for x, y in xy[sel]],
                                          facecolor=col, lw=0, zorder=3))
    # curated units only: bombcell noise and the ISI-excluded units are 'EXCL'
    # and are dropped here as everywhere else in the paper
    cells = F.CELLXYZ[(F.CELLXYZ.mouse == mo) & (F.CELLXYZ.day == dy)
                      & F.CELLXYZ.macro.isin(['MEC', 'VIS'])
                      & (F.CELLXYZ.cell_class_of1 != 'EXCL')].dropna(subset=['coord_probe_y'])
    pu, pv = cells.coord_probe_x.values.astype(float), cells.coord_probe_y.values.astype(float)
    ex = np.array([rr.excess.get(c, np.nan) for c in cells.cluster_id], float)
    # locked cells have no agreement and are drawn as chance, as on the slice
    ex = np.where(np.isfinite(ex), ex, 0.)
    sc = ax.scatter(pu, pv, c=ex, s=7.0, cmap=AGREE_CMAP,
                    norm=TwoSlopeNorm(vcenter=0., vmin=-vlim, vmax=vlim),
                    alpha=.95, lw=.25, edgecolors='0.45', zorder=5)
    cid_ = cells.cluster_id.values.astype(int)
    for c_, k_ in marks:
        j_ = np.flatnonzero(cid_ == int(c_))
        if len(j_):
            ax.scatter(pu[j_], pv[j_], s=46, facecolor='none', edgecolor='k', lw=.9, zorder=6)
            ax.annotate(str(k_), (pu[j_[0]], pv[j_[0]]), xytext=(5, 0), textcoords='offset points',
                        fontsize=6.5, weight='bold', va='center', zorder=7)
    ya = xy[act, 1]
    ax.set_xlim(-60, 3 * 250 + 32 + 60)
    ax.set_ylim(ya.min() - 150, ya.max() + 150)
    ax.set_aspect('equal', adjustable='box'); ax.set_anchor('C')
    ax.set_xticks([16 + 250 * k for k in range(4)]); ax.set_xticklabels(['1', '2', '3', '4'])
    ax.set_xlabel('shank', fontsize=6.5, labelpad=1)
    ax.set_ylabel('µm from probe tip', fontsize=6.5, labelpad=1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax._cells_x = pu
    ax._cells = dict(cid=cid_, pv=pv, macro=cells.macro.values,
                     dv=cells.coord_SCs_y.values.astype(float))
    return sc


def example(fig, gs, mo, dy, letter, vlim, marks=()):
    B, tr, rr = blocks(mo, dy)
    n_tr = B['MEC']['M'].shape[1]
    gg = gs.subgridspec(1, 5, width_ratios=[1.2, .24, .9, .6, .8], wspace=.16)
    axh = None
    HR = [.05, 1]
    gR = gg[0].subgridspec(2, 2, height_ratios=HR, hspace=.025, wspace=.06,
                           width_ratios=[B['VIS']['M'].shape[0], B['MEC']['M'].shape[0]])
    ax0 = None
    # dorsal -> ventral across the whole raster: visual cortex sits above MEC
    # on the probe, so its block comes first
    for j, name in enumerate(('VIS', 'MEC')):
        b = B[name]
        axs = fig.add_subplot(gR[0, j])
        axs.imshow(np.repeat(b['strip'][None, :, None], 3, axis=2), aspect='auto',
                   interpolation='nearest', vmin=0, vmax=1)
        axs.set_xticks([]); axs.set_yticks([])
        for sp in axs.spines.values():
            sp.set_visible(False)
        # VIS is often a narrow block: its header hangs left, MEC's right
        axs.set_title(f'{name} {b["n"]}\n{b["n_lock"]} locked '
                      f'({100 * b["n_lock"] / b["n"]:.0f}%)', fontsize=6.3, pad=2,
                      color='0.2', loc='left' if j == 0 else 'right', linespacing=1.1)
        ax = fig.add_subplot(gR[1, j], **({} if ax0 is None else {'sharey': ax0}))
        ax0 = ax0 or ax
        ax.imshow(np.ma.masked_invalid(b['M'].T), aspect='auto', interpolation='nearest',
                  cmap=TAC, norm=BoundaryNorm([-.5, .5, 1.5], 2),
                  extent=[0, b['M'].shape[0], n_tr, 1])
        for t_ in tr:
            ax.axhline(t_ + 1, color='0.1', lw=.5, ls='--', alpha=.7, zorder=4)
        ax.set_xticks([])
        ax.set_xlabel(f'{name} cells', fontsize=6.8, labelpad=2)
        if j == 0:
            ax.set_ylabel('Trial', fontsize=8, labelpad=1)
            ax.tick_params(labelsize=6.5)
            axs.set_ylabel('region', fontsize=5.8, rotation=0, ha='right', va='center',
                           labelpad=2)
            axs.annotate(f'M{mo} D{dy}', (0, 1), xycoords='axes fraction',
                         xytext=(0, 12), textcoords='offset points', fontsize=7.6,
                         ha='left', va='bottom')
            _p = gR[0, 0].get_position(fig)
            fig.text(_p.x0 - .04, _p.y1 + .02, letter, fontsize=10, weight='bold',
                     va='bottom', ha='right')
        else:
            ax.tick_params(labelleft=False, left=False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    gP = gg[1].subgridspec(2, 1, height_ratios=HR, hspace=.025)
    axp = fig.add_subplot(gP[1], sharey=ax0)
    y = np.arange(1, n_tr + 1)
    for name in ('VIS', 'MEC'):
        axp.plot(B[name]['pc'], y, color=LCOL[name], lw=.8, zorder=3)
    axp.axvline(0, color='0.6', lw=.6)
    for t_ in tr:
        axp.axhline(t_ + 1, color='0.1', lw=.5, ls='--', alpha=.7, zorder=4)
    axp.set_xticks([]); axp.tick_params(labelleft=False, left=False)
    axp.set_xlabel('PC1\n(z)', fontsize=7, labelpad=2)
    for sp in axp.spines.values():
        sp.set_visible(False)
    rmv = np.corrcoef(B['MEC']['pc'], B['VIS']['pc'])[0, 1]
    axp.set_title(f'r = {rmv:+.2f}', fontsize=6.3, pad=2, color='0.2')

    axsl = fig.add_subplot(gg[2])
    sc = draw_probe(axsl, mo, dy, rr, vlim, marks)
    axsl.apply_aspect()
    # validation of the depth axis: the shank's LFP spectra on the slice's
    # depth axis, with the physiological landmarks of Figure S15 drawn across
    # both, and the histological section holding the shank's DiI track
    axlf = fig.add_subplot(gg[3], sharey=axsl)
    _yl = axsl.get_ylim()
    lfp_panel(axlf, axsl, mo, dy)
    axsl.set_ylim(_yl)
    _p, _q = axsl.get_position(), axlf.get_position()
    axlf.set_position([_p.x1 + .026, _p.y0, _q.width, _p.height])
    if letter == 'A':
        from matplotlib.lines import Line2D
        axlf.legend(handles=[Line2D([], [], color='#e66101', lw=1.3, label='LFP: VIS / MEC'),
                             Line2D([], [], color='#5e3c99', lw=1.1, label='LFP: leaves cortex'),
                             Line2D([], [], color='k', ls='--', lw=.8, label='atlas: VIS / MEC')],
                    loc='upper left', bbox_to_anchor=(-.05, -.16), fontsize=5.3, frameon=False,
                    handlelength=1.4, borderaxespad=0, labelspacing=.25)
    import anatomy_rules as AR
    sh = int(AR.shank_of(mo, np.median(F.CELLXYZ[(F.CELLXYZ.mouse == mo) & (F.CELLXYZ.day == dy)].coord_SCs_x)))
    got = hist_section(mo, sh) if mo in HIST else None
    if got is not None:
        img, sec, um = got
        axh = fig.add_subplot(gg[4])
        axh.imshow(img, aspect='equal')
        axh.apply_aspect()
        _h, _l = axh.get_position(), axlf.get_position()
        axh.set_position([_l.x1 + .012, _h.y0, _h.width, _h.height])
        axh.plot([img.shape[1] * .08, img.shape[1] * .08 + 500 / um], [img.shape[0] * .94] * 2, color='w', lw=1.5)
        axh.text(img.shape[1] * .08, img.shape[0] * .89, '500 um', color='w', fontsize=5)
        axh.set_xticks([]); axh.set_yticks([])
        axh.set_title(f'histology, section {sec}\n(shank {sh + 1})', fontsize=6.3, pad=2, linespacing=1.1)
        from matplotlib.patches import Patch
        axh.legend(handles=[Patch(color='#3b6cff', label='DAPI'), Patch(color='#ff2a2a', label='DiI')],
                   loc='upper center', bbox_to_anchor=(.5, -.01), ncol=2, fontsize=5.5, frameon=False,
                   handlelength=.9, handleheight=.9, columnspacing=.9, handletextpad=.4, borderaxespad=0)
    print(f'M{mo} D{dy}: MEC {B["MEC"]["n_lock"]}/{B["MEC"]["n"]} locked, '
          f'VIS {B["VIS"]["n_lock"]}/{B["VIS"]["n"]} locked, {len(tr)} transitions, '
          f'{n_tr} trials, r(MEC PC1, VIS PC1) = {rmv:+.3f}')
    fig._probe_axes = (axsl, axlf, axh)          # for callers that add to the row
    return sc


if __name__ == '__main__':
    from scipy.stats import wilcoxon
    ex = REG[REG.set_index(['mouse', 'day']).index.isin(SESSIONS)].excess.dropna()
    vlim = float(max(abs(np.percentile(ex, 2)), abs(np.percentile(ex, 98))))
    fig = plt.figure(figsize=(10.4, 11.6))
    outer = fig.add_gridspec(4, 1, height_ratios=[1, 1, .62, .62], hspace=.42,
                             left=.06, right=.985, top=.96, bottom=.05)
    for i, (mo, dy) in enumerate(SESSIONS):
        row = outer[i].subgridspec(1, 2, width_ratios=[1.05, 1], wspace=.14)
        picks = pick_followers(mo, dy, n=6, by_agree=BY_AGREE.get((mo, dy), ()), force=FORCE.get((mo, dy)),
                               min_vis=3, dorsal=True)
        print(f'M{mo}D{dy} examples: ' + ', '.join(f'{k + 1}:{st} {c} r={a:+.2f} dv={d:.0f}'
                                                  for k, (c, st, a, d) in enumerate(picks)))
        sc = example(fig, row[0], mo, dy, 'AC'[i], vlim, marks=[(c, k + 1) for k, (c, *_) in enumerate(picks)])
        maps_panel(fig, row[1], mo, dy, picks, 'BD'[i])
    # ---- E-G: is the state shared? ---------------------------------------
    SS, CC, TT = shared_state(exclude=True)
    # F-I use only sessions whose cells reach >= MIN_EXTENT from the border on
    # both sides: near the border an assignment error between the structures
    # would masquerade as sharing. TT's per-session arrays follow SS's order.
    EXT = session_extent()
    keep = np.array([EXT.extent.get((m, d), 0) >= MIN_EXTENT for m, d in zip(SS.mouse, SS.day)])
    assert TT['MEC>VIS'].shape[0] == len(SS)
    TT = {k: (v[keep] if k.endswith(('_n',)) or v.ndim >= 2 else v[keep]) for k, v in TT.items()}
    SS = SS[keep].reset_index(drop=True)
    _ok = set(zip(SS.mouse, SS.day))
    CC = CC[[(m, d) in _ok for m, d in zip(CC.mouse, CC.day)]]
    print(f'F-I: {len(SS)} sessions reach >= {MIN_EXTENT:.0f} um on both sides')
    gB = outer[3].subgridspec(1, 4, width_ratios=[1, 1, 1.05, 1.25], wspace=.42)
    lags = np.arange(-WIN, WIN)
    axE0 = None
    for j, trg in enumerate(('MEC', 'VIS')):
        ax = fig.add_subplot(gB[j], **({} if axE0 is None else {'sharey': axE0}))
        axE0 = axE0 or ax
        oth = 'VIS' if trg == 'MEC' else 'MEC'
        Nm = TT[f'{trg}>{oth}_null'].mean(0)            # shifts x lags, session mean
        lo, hi = np.percentile(Nm, [2.5, 97.5], axis=0)
        ax.fill_between(lags, lo, hi, facecolor='none', hatch='.....', edgecolor='0.75', lw=0,
                        zorder=1)
        for k in (trg, oth):
            X = TT[f'{trg}>{k}']
            m, se = X.mean(0), X.std(0) / np.sqrt(len(X))
            ax.fill_between(lags, m - se, m + se, color=LCOL[k], alpha=.25, lw=0,
                            edgecolor='none', zorder=2)
            ax.plot(lags, m, color=LCOL[k], lw=1.3, zorder=3)
        ax.axvline(-.5, color='0.3', lw=.6, ls='--'); ax.axhline(0, color='0.5', lw=.6, ls=':')
        # the other structure's step (trials 0-4 against the pre-switch
        # baseline the traces are already referred to), as a share of the
        # triggering structure's own, and against the shifted null
        post = slice(WIN, WIN + 5)
        st_o = TT[f'{trg}>{oth}'][:, post].mean(1); st_t = TT[f'{trg}>{trg}'][:, post].mean(1)
        nul = Nm[:, post].mean(1)
        p = (np.sum(nul >= st_o.mean()) + 1) / (len(nul) + 1)
        _ps = (f'p < {1 / (len(nul) + 1):.1g}' if p <= 1 / (len(nul) + 1) + 1e-12
               else f'p = {p:.2g}')
        ax.set_title(f'at {trg} switches\n{oth} steps {100 * st_o.mean() / st_t.mean():.0f}% '
                     f'as far, {_ps}', fontsize=7, pad=3, linespacing=1.15)
        print(f'E {trg} switches: {int(TT[f"{trg}_n"].sum())} switches in {len(st_o)} sessions; '
              f'{trg} step {st_t.mean():.3f}, {oth} step {st_o.mean():.3f} '
              f'({100 * st_o.mean() / st_t.mean():.0f}%), null 97.5% {np.percentile(nul, 97.5):.3f}, p {p:.3g}')
        ax.set_xlabel(f'trials from {trg} switch', fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.spines[['top', 'right']].set_visible(False)
        if j == 0:
            ax.set_ylabel('change in fraction of cells\nin the new state', fontsize=7.5)
            ax.fill_between([], [], [], facecolor='none', hatch='.....', edgecolor='0.75',
                            lw=0, label='time-shift null')
            ax.legend(fontsize=6, frameon=False, loc='upper left', handlelength=1.2)
            ax.text(-.36, 1.12, 'G', transform=ax.transAxes, fontsize=10, weight='bold')
        else:
            ax.tick_params(labelleft=False)

    # F: shared as far as each axis's own reliability allows
    ax = fig.add_subplot(gB[2])
    sig = SS.p_mv < .05
    ax.plot([0, 1], [0, 1], color='0.6', lw=.7, ls='--', zorder=1)
    ax.axhline(0, color='0.5', lw=.6, ls=':')
    ax.scatter(SS.ceiling[sig], SS.r_mv[sig], s=16, color='0.15', lw=0, zorder=3)
    ax.scatter(SS.ceiling[~sig], SS.r_mv[~sig], s=16, facecolor='w', edgecolor='0.15',
               lw=.8, zorder=3)
    for L_, (mo, dy) in zip('AC', SESSIONS):
        r_ = SS[(SS.mouse == mo) & (SS.day == dy)]
        if len(r_):
            ax.annotate(L_, (r_.ceiling.iloc[0], r_.r_mv.iloc[0]), xytext=(3, -7),
                        textcoords='offset points', fontsize=6, weight='bold', color='0.3')
    hc = SS.ceiling >= .5
    ratio = (SS.r_mv / SS.ceiling)[hc]
    ax.set_xlim(0, 1); ax.set_ylim(-.6, 1)
    ax.set_xlabel('reliability ceiling', fontsize=7.5)
    ax.set_ylabel('r (MEC PC1, VIS PC1)', fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(f'{int(sig.sum())} of {len(SS)} beyond shift null\nceiling >= 0.5: r / ceiling '
                 f'median {ratio.median():.2f} (n = {hc.sum()})', fontsize=7, pad=3, linespacing=1.15)
    ax.text(-.3, 1.12, 'H', transform=ax.transAxes, fontsize=10, weight='bold')
    print(f'F: sig {int(sig.sum())}/{len(SS)}; ceiling>=.5: n {hc.sum()}, r/ceiling median '
          f'{ratio.median():.3f}, below .5: {int((ratio < .5).sum())} '
          f'{[f"M{a}D{b}" for a, b in zip(SS.mouse[hc][ratio < .5], SS.day[hc][ratio < .5])]}; '
          f'ceiling<.5: n {(~hc).sum()}, sig {int(sig[~hc].sum())}')

    # G: census of cells against each axis
    ax = fig.add_subplot(gB[3])
    bars = [('VIS', 'MEC'), ('VIS', 'VIS'), ('MEC', 'MEC'), ('MEC', 'VIS')]
    cats = [('follows', '0.20'), ('opposes', '0.55'), ('neither', '0.86'), ('locked', 'w')]
    for y_, (st, k) in enumerate(bars):
        c = CC[CC.structure == st]
        v = c[f'cat_{k}'].where(~c.locked, 'locked').value_counts(normalize=True)
        x0 = 0.
        for cat, col in cats:
            w = float(v.get(cat, 0))
            ax.barh(y_, w, left=x0, color=col, edgecolor='0.4', lw=.4, height=.7,
                    hatch='////' if cat == 'locked' else None)
            if cat == 'follows':
                ax.text(x0 + w / 2, y_, f'{100 * w:.0f}%', ha='center', va='center',
                        fontsize=6, color='w')
            x0 += w
        print(f'G {st} cells vs {k} axis (n={len(c)}): ' +
              ', '.join(f'{cat} {100 * float(v.get(cat, 0)):.1f}%' for cat, _ in cats))
    ax.set_yticks(range(4))
    ax.set_yticklabels([f'{st} cells,\n{"own" if st == k else k} axis' for st, k in bars],
                       fontsize=6.3)
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel('fraction of cells', fontsize=7.5)
    ax.tick_params(axis='x', labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(handles=[matplotlib.patches.Patch(facecolor=col, edgecolor='0.4', lw=.4,
                                                hatch='////' if cat == 'locked' else None,
                                                label=cat) for cat, col in cats],
              fontsize=6, frameon=False, ncol=4, loc='lower center',
              bbox_to_anchor=(.4, 1.0), handlelength=1.2, columnspacing=.8)
    ax.text(-.42, 1.12, 'I', transform=ax.transAxes, fontsize=10, weight='bold')

    # ---- H: state responsiveness across the VIS-MEC border ------------------
    # One axis through both structures: each cell at its 3D atlas distance to
    # the other structure, visual cells to the left of the border and MEC cells
    # to the right, read against both structures' axes. Single-shank sessions,
    # so distance from the border is distance along one shank.
    D, _ = border_agreement()
    # I keeps every cell of every single-shank session (no extent rule, no
    # near-border exclusion), so the lines run across the whole range
    gH = outer[2].subgridspec(1, 3, width_ratios=[.62, 1.45, .62], wspace=.3)
    # ---- E: agreement within and across the structures ---------------------
    axE = fig.add_subplot(gH[0])
    _ma = np.where(D.structure == 'MEC', D.excess_own, D.excess)
    _va = np.where(D.structure == 'VIS', D.excess_own, D.excess)
    _sess = D.mouse.astype(str) + '_' + D.day.astype(str)
    groups = [('VIS', 'VIS', _va), ('VIS', 'MEC', _ma), ('MEC', 'MEC', _ma), ('MEC', 'VIS', _va)]
    data_ = [vals[(D.structure == st).values] for st, _, vals in groups]
    bp_ = axE.boxplot(data_, positions=[0, 1, 2.4, 3.4], widths=.6, showfliers=False, patch_artist=True,
                      medianprops=dict(color='k', lw=1.3), whiskerprops=dict(lw=.7, color='0.35'),
                      capprops=dict(lw=.7, color='0.35'))
    for patch, (st, axn, _) in zip(bp_['boxes'], groups):
        patch.set_facecolor('#c9e3cd' if axn == 'VIS' else '0.78'); patch.set_edgecolor('0.35'); patch.set_lw(.7)
    axE.axhline(0, color='0.5', lw=.6, ls=':')
    axE.set_xticks([0, 1, 2.4, 3.4]); axE.set_xticklabels(['own', 'MEC', 'own', 'VIS'], fontsize=6.5)
    for x_, t_ in ((.5, 'VIS cells'), (2.9, 'MEC cells')):
        axE.text(x_, -.17, t_, transform=axE.get_xaxis_transform(), ha='center', fontsize=7,
                 color=LCOL['VIS'] if t_.startswith('VIS') else '0.15')
    axE.set_ylabel('agreement above chance', fontsize=7.5)
    axE.set_title('against each structure\'s PC1', fontsize=7, pad=13)
    axE.tick_params(axis='y', labelsize=6.5); axE.spines[['top', 'right']].set_visible(False)
    _pp = {}
    for (st, ax1, v1), (_, ax2, v2) in ((groups[0], groups[1]), (groups[2], groups[3])):
        m_ = (D.structure == st).values
        sm = pd.DataFrame(dict(s=_sess[m_], a=v1[m_], b=v2[m_])).groupby('s').median()
        _pp[st] = wilcoxon(sm.a, sm.b).pvalue
        print(f'E {st} cells: own {np.median(v1[m_]):.3f} vs other {np.median(v2[m_]):.3f}, '
              f'within-session p = {_pp[st]:.2g} ({len(sm)} sessions)')
    lo_, hi_ = axE.get_ylim()
    for x_, st in ((.5, 'VIS'), (2.9, 'MEC')):
        axE.text(x_, hi_, f'p = {_pp[st]:.2g}', ha='center', va='bottom', fontsize=6)
    axE.text(-.42, 1.08, 'E', transform=axE.transAxes, fontsize=10, weight='bold')
    ax = fig.add_subplot(gH[1])
    xs = np.where(D.structure == 'VIS', -D.dist, D.dist)
    agree = {'MEC': np.where(D.structure == 'MEC', D.excess_own, D.excess),
             'VIS': np.where(D.structure == 'VIS', D.excess_own, D.excess)}
    # Sessions differ in overall agreement AND in how far their cells sit from
    # the border, so a pooled curve mixes the two: drawn raw it peaks at the
    # border on both sides although no within-session test finds that. Each
    # session's mean (per structure and axis) is removed and the grand mean
    # added back, so the curves show only how agreement changes with distance
    # within a session, on the scale of agreement above chance.
    _key = D.mouse.astype(str) + '_' + D.day.astype(str) + '_' + D.structure
    for k_ in agree:
        v_ = pd.Series(agree[k_])
        g_ = v_.groupby(_key.values)
        agree[k_] = (v_ - g_.transform('mean') + v_.groupby(D.structure.values).transform('mean')).values
    # Fixed 200 um bins from the border outward, drawn where a bin
    # holds at least MIN_BIN cells, with every cell as a faint point beneath:
    # equal-count bins put the nearest point at the median of a wide first bin
    # and left the region next to the border looking empty.
    MIN_BIN = 5
    edges_ = np.arange(0, 1600 + 1, 200)
    for side in ('VIS', 'MEC'):
        m = (D.structure == side).values
        dist = np.abs(xs[m])
        b = np.digitize(dist, edges_) - 1
        for axis_ in ('VIS', 'MEC'):
            y_all = agree[axis_][m]
            g_ = pd.DataFrame(dict(b=b, y=y_all, x=xs[m])).groupby('b')
            st_ = g_.agg(n=('y', 'size'), mu=('y', 'mean'), sd=('y', 'std'))
            st_ = st_[(st_.n >= MIN_BIN) & (st_.index >= 0) & (st_.index < len(edges_) - 1)]
            xc = (edges_[st_.index] + 100) * (-1 if side == 'VIS' else 1)
            se = st_.sd / np.sqrt(st_.n)
            ax.fill_between(xc, st_.mu - se, st_.mu + se, color=LCOL[axis_], alpha=.22, lw=0,
                            edgecolor='none', zorder=2)
            ax.plot(xc, st_.mu, color=LCOL[axis_], lw=1.4, marker='o', ms=2.5, zorder=3)
    ax.set_xlim(-1300, 1500)
    ax.axvline(0, color='0.4', lw=.8, ls='--')
    ax.axhline(0, color='0.5', lw=.6, ls=':')
    ax.set_ylim(-.03, .16)
    ax.set_xlabel('distance from the VIS-MEC border (µm)', fontsize=7.5)
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    ax.text(.02, .97, 'visual cortex cells', transform=ax.transAxes, fontsize=7, color='0.3',
            va='top')
    ax.text(.98, .97, 'MEC cells', transform=ax.transAxes, fontsize=7, color='0.3',
            va='top', ha='right')
    # within-session rank correlation with distance, each side and each axis,
    # printed in that side's half of the panel in the axis's colour
    for side, other, xa, ha in (('VIS', 'MEC', .02, 'left'), ('MEC', 'VIS', .98, 'right')):
        d_ = D[D.structure == side]
        for k, (y_, axis_) in enumerate((('excess_own', side), ('excess', other))):
            S_ = border_stats(d_, y_)
            ax.text(xa, .03 + .075 * k, f'vs {axis_} PC1: within-session rho {S_["rho_ws"]:+.3f}, '
                    f'p = {S_["p_ws"]:.2g}', transform=ax.transAxes, fontsize=5.9,
                    color=LCOL[axis_], ha=ha)
            print(f'H {side} cells vs {axis_} PC1: within-session rho {S_["rho_ws"]:+.4f} '
                  f'p {S_["p_ws"]:.3g} ({S_["neg_ws"]}/{S_["n_ws"]} < 0), n {S_["n_cells"]}')
    ax.set_title(f'agreement with each structure\'s PC1 across the border ({len(D)} cells, '
                 f'{D.groupby(["mouse", "day"]).ngroups} single-shank sessions, session means removed)',
                 fontsize=6.8, loc='left', pad=4)
    ax.set_ylabel('agreement\n(within session)', fontsize=7)
    ax.text(-.09, 1.08, 'F', transform=ax.transAxes, fontsize=10, weight='bold')

    # the shared key, beside H: the agreement scale (common to the four
    # slices), the PC1 lines and the region strip greys
    _pk = gH[2].get_position(fig)
    cax = fig.add_axes([_pk.x0 + .02, _pk.y1 - .02, _pk.width * .8, .008])
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
    cb.set_label('agreement above chance (A-D slices)', fontsize=7, labelpad=2)
    h = ([Line2D([], [], color=LCOL[k], lw=1.2, label=f'{k} (PC1 lines, F)') for k in ('MEC', 'VIS')]
         + [matplotlib.patches.Patch(facecolor=str(g), edgecolor='0.6', lw=.4, label=k)
            for k, g in (('MEC superficial', GREY['MEC sup']), ('MEC deep', GREY['MEC deep']),
                         ('visual cortex', GREY['VIS']))]
         + [Line2D([], [], marker='o', ls='', markerfacecolor='w', markeredgecolor='0.45', ms=3, label='locked (drawn as chance)')])
    fig.legend(handles=h, loc='upper left', bbox_to_anchor=(_pk.x0 + .01, _pk.y1 - .065), ncol=1,
               fontsize=6.6, frameon=False, handlelength=1.4, columnspacing=1.2)
    tmp = OUT + '.partial'
    fig.savefig(tmp, format='pdf', dpi=300); os.replace(tmp, OUT)
    print('wrote', OUT)

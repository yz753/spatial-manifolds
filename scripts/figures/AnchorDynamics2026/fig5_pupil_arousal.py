"""Figure 6 — the anchoring state tracks attentional engagement.

    A   camera and IR illumination schematic, the recording arrangement
    B   the pupillometry drawing, landmarks and radius definition
    C   the eye itself, seven moments in one traversal, for one trial of each state
    D   mean pupil radius against track position, one curve per state block
    E   the population anchoring raster -- every entorhinal cell, every trial
    F   PC1 of that raster, the population state as one number per trial
    G   the same pupil radius per TRIAL, as a heatmap, on the trial axis of E and F
    H   pupil radius by state, paired within session across all included sessions
    I   dilation by position, population-level, aligned to D's x-span (both plot
        the same track-position axis, so a feature under D sits at the same
        position in I)

E, F and G share the trial axis, so a block of anchored trials in the raster can
be read straight across into the pupil rows it corresponds to; D and I share the
position axis explicitly (I's geometry is set to match D's after both are drawn,
since they come from different parent gridspecs and would not align otherwise).

An earlier version took each track position from whichever trial happened to be
most median at that position, so a row mixed seven trials and was not a journey
down the track at all. Here each row is ONE trial, so the seven frames are
consecutive moments in a single traversal.

Choosing the trial: within each state's longest block, the trial whose
position-binned pupil radius profile is closest (RMS) to that block's mean
profile, among trials with usable frames at all seven positions. That is the
most representative single traversal, not the most extreme one.

The raster comes from the canonical label file, not from
eye_anchoring_trials.csv, which is a derived copy -- for this session the two
agree exactly (r = 1.000 over 107 trials), but only the label file carries the
per-cell matrix and PC1 that B and C need.

VIDEO GEOMETRY differs between sessions and gets this wrong silently:
  * full-frame videos (1440x1080) need the crop origin from all_eye_crops.csv;
    NWB landmark coordinates are relative to that crop.
  * '*_eye_zone.avi' files are ALREADY cropped, but to a tighter box than the
    table lists, so NWB coordinates need a further offset. That offset is solved
    from the data -- match the dark-pupil centroid to the NWB centroid over
    sampled frames -- and verified by eye before use. For M28 D18 it is
    (-30.4, -31.4) with an SD of 1.6 px across frames.
"""
import os, sys, json, time, warnings, glob
import cv2, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import median_filter
from scipy.stats import wilcoxon
import pynapple as nap
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import load_session_labels
_g = {}
for _c in json.load(open('/Users/harryclark/Documents/spatial-manifolds/scripts/'
                         'figures/AnchorDynamics2026/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})
plt.rcParams['font.family'] = 'Arial'

FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
VROOT = '/Volumes/INCR-NolanLab/ActiveProjects/Harry/EphysNeuropixelData/vr'
# Read lazily: the frames are cached per session, so the figure rebuilds without
# the lab volume mounted. Only a session whose frames are NOT cached needs it.
CROPS_PATH = ('/Volumes/INCR-NolanLab/ActiveProjects/Chris/Cohort12/derivatives/'
              'labels/deeplabcut/all_eye_crops.csv')


APPARATUS_SCALE = {'camera_and_IR_mouse.pdf': 1.10,
                   'pupilometry.pdf': 1.22}
# Inches to the right of the box centre. The enlargements above push each
# drawing past its own box, and A's floor rectangle met B's frame with no gap
# at all; B has 0.63 in of empty page to its right and A has none to its left,
# so the clearance is taken from B's side.
APPARATUS_DX = {'pupilometry.pdf': 0.20}


def _place_apparatus(pdf_path, boxes, tightbbox, figsize):
    """Draw the vector apparatus PDFs into the axes left empty for them.

    matplotlib cannot embed a PDF, and rasterising line art at figure scale
    throws away the detail these drawings exist for, so they are composited into
    the finished page with PyMuPDF. Positions come from the axes themselves
    rather than being hardcoded, corrected for the tight-bbox crop, so the
    artwork follows the layout if the gridspec changes.
    """
    import pymupdf
    fw, fh = figsize
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    for pos, name in boxes:
        src_path = os.path.join(FIG, name)
        if not os.path.exists(src_path):
            print(f'  ! {name} not found, box left blank')
            continue
        src = pymupdf.open(src_path)
        r = src[0].rect
        x0, y0 = pos.x0 * fw, pos.y0 * fh          # axes box, figure inches
        x1, y1 = pos.x1 * fw, pos.y1 * fh
        px0 = (x0 - tightbbox.x0) * 72             # into the cropped page,
        px1 = (x1 - tightbbox.x0) * 72             # y measured down from the top
        py0 = (tightbbox.y1 - y1) * 72
        py1 = (tightbbox.y1 - y0) * 72
        bw, bh = px1 - px0, py1 - py0
        # Fitting inside the box leaves whichever dimension is not limiting
        # empty, and the pupillometry drawing is much wider than its box is
        # (aspect 1.37 against a taller slot), so a third of its height was
        # blank. Each drawing gets a modest enlargement past the box; they sit
        # in empty axes with no ticks or spines, so a little overflow costs
        # nothing. The pupillometry one is allowed more because it is the one
        # with the slack.
        sc = min(bw / r.width, bh / r.height)      # keep aspect, centre in box
        sc *= APPARATUS_SCALE.get(name, 1.0)
        w, h = r.width * sc, r.height * sc
        cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
        cx += APPARATUS_DX.get(name, 0.0) * 72
        page.show_pdf_page(pymupdf.Rect(cx - w / 2, cy - h / 2,
                                        cx + w / 2, cy + h / 2), src, 0)
        src.close()
        print(f'  placed {name} ({w/72:.2f} x {h/72:.2f} in)')
    doc.saveIncr()
    doc.close()


def crops():
    if not os.path.exists(CROPS_PATH):
        raise RuntimeError(f'{CROPS_PATH} not found -- is /Volumes/INCR-NolanLab '
                           'mounted? (only needed when frames are not cached)')
    return pd.read_csv(CROPS_PATH)
SHOW_POS = [0, 30, 60, 90, 120, 150, 170]
POS_TOL = 5.0
LM = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']
LM_COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8',
             '#f58231', '#42d4f4', '#f032e6', '#fabed4']
ZOOM = 46
# sessions where the video is already cropped and needs a further offset
# Solved from the data (dark-pupil centroid vs NWB centroid) and checked by eye.
# Both come out near (-31, -31), so the eye_zone crop origin is effectively fixed.
EYE_ZONE_OFFSET = {(28, 18): (-30.4, -31.4), (29, 17): (-31.0, -31.7)}


TRIALS_CSV = ('/Users/harryclark/Documents/spatial-manifolds/data/eye_anchoring/'
              'eye_anchoring_trials.csv')
WIN, MIN_RUN = 5, 3      # transition window, and how long each state must hold


def eye_trials():
    """Per-trial eye table with the notebook's inclusion criteria applied.

    `load_trials` is exec'd out of figure5_pupil_arousal.ipynb rather than
    reimplemented: its thresholds were calibrated by subsampling (MIN_PER_CLASS
    in particular) and a second copy here would drift from them silently.
    """
    nb = json.load(open(f'{FIG}/figure5_pupil_arousal.ipynb'))
    src = next(''.join(c['source']) for c in nb['cells']
               if c['cell_type'] == 'code' and 'def load_trials' in ''.join(c['source']))
    g = dict(pd=pd, np=np, TRIALS_CSV=TRIALS_CSV)
    exec(compile(src, '<eye_nb>', 'exec'), g)
    return g['load_trials'](verbose=False)[0]


PROFILE_CACHE = ('/Users/harryclark/Documents/spatial-manifolds/data/eye_anchoring/'
                 'eye_position_profiles_gated.npz')
NPOS = 50                # 4 cm position bins over the 200 cm track


def position_profiles(E, rebuild=False):
    """Per-session z(pupil radius) against track position, split by state.

    NOT read from eye_anchoring_profiles.npz: that file predates the absolute
    gate, and its per-state trial counts no longer match the labels in use
    (M25D23 splits 99/102 there against 113/88 now), so its profiles are for a
    different partition of the trials. Rebuilt here from the current labels and
    cached, since it costs one NWB load per session.

    The radius is z-scored within session over RUNNING samples before binning,
    so a session with a larger pupil or a closer camera cannot dominate, and
    each session contributes one profile per state however many trials it has.
    """
    if os.path.exists(PROFILE_CACHE) and not rebuild:
        z = np.load(PROFILE_CACHE)
        return z['anch'], z['non'], z['keys']
    A, N, K = [], [], []
    for (mo, dy), g in E.groupby(['mouse', 'day']):
        try:
            S = load_session(int(mo), int(dy))
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        ok = S['good'] & (S['spd'] >= 3.0)
        if ok.sum() < 1000:
            continue
        r = S['rad'].astype(float)
        zr = (r - np.nanmean(r[ok])) / np.nanstd(r[ok])
        pb = np.clip((S['pos'] / (200.0 / NPOS)).astype(int), 0, NPOS - 1)
        st = dict(zip(g.trial.astype(int), (g.frac_anch > .5).values))
        lab = np.array([st.get(int(t), np.nan) if t > 0 else np.nan
                        for t in S['tr']], dtype=float)
        out = {}
        for nm, want in (('a', 1.0), ('n', 0.0)):
            # not averaging like the manuscript (with session first then across sessions)
            m = ok & (lab == want)
            prof = np.full(NPOS, np.nan)
            if m.sum() > 200:
                cnt = np.bincount(pb[m], minlength=NPOS).astype(float)
                tot = np.bincount(pb[m], weights=zr[m], minlength=NPOS)
                prof = np.where(cnt > 20, tot / np.maximum(cnt, 1), np.nan)
            out[nm] = prof
        if np.isfinite(out['a']).sum() > NPOS * .8 and \
           np.isfinite(out['n']).sum() > NPOS * .8:
            A.append(out['a']); N.append(out['n']); K.append(f'M{mo}D{dy}')
            print(f'  M{mo}D{dy}: profiles built', flush=True)
    A, N, K = np.array(A), np.array(N), np.array(K)
    np.savez_compressed(PROFILE_CACHE, anch=A, non=N, keys=K)
    print(f'built {PROFILE_CACHE}: {len(K)} sessions')
    return A, N, K


def transition_segments(E, col='z_rad'):
    """Per-session mean of `col` in a window around each state transition.

    A transition counts only if the old state held for MIN_RUN trials and the
    new one holds MIN_RUN as well, so flicker cannot contaminate the window.
    Sessions are averaged first and then pooled: pooling raw transitions would
    weight a session by how often it switched, and switch rate varies
    several-fold between sessions.
    """
    out = {}
    for into, lab in ((True, 'anchored'), (False, 'non-anchored')):
        per, n_tr = [], 0
        for _, g in E.groupby(['mouse', 'day']):
            g = g.sort_values('trial')
            st, v = g.anch.values, g[col].values
            idx = [i for i in range(MIN_RUN, len(st) - MIN_RUN + 1)
                   if st[i] == into and st[i - 1] != into
                   and all(st[i - k] == (not into) for k in range(1, MIN_RUN + 1))
                   and all(st[i + k] == into for k in range(MIN_RUN))]
            segs = [v[i - WIN:i + WIN + 1] for i in idx
                    if i - WIN >= 0 and i + WIN + 1 <= len(v)]
            if segs:
                per.append(np.nanmean(segs, axis=0)); n_tr += len(idx)
        out[lab] = (np.array(per), n_tr)
    return out


def find_video(mo, dy):
    for pat in (f'{VROOT}/M{mo}_D{dy}_*/*eye_zone.avi',
                f'{VROOT}/M{mo}_D{dy}_*/sub-*video.avi'):
        g = glob.glob(pat)
        if g:
            return g[0]
    return None


def load_session(mo, dy):
    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    t = np.asarray(beh['eye_x'].index)
    X = np.array([np.asarray(beh[f'eye_{k}_x'].values) for k in LM])
    Y = np.array([np.asarray(beh[f'eye_{k}_y'].values) for k in LM])
    lik = np.nanmean([np.asarray(beh[f'eye_{k}_likelihood'].values) for k in LM], axis=0)
    cx, cy = X.mean(0), Y.mean(0)
    rad = np.hypot(X - cx, Y - cy).mean(0)
    at = lambda d: np.asarray(d.values)[np.searchsorted(np.asarray(d.index), t)
                                        .clip(0, len(d) - 1)]
    pos, spd, trn = at(beh['P']), at(beh['S']), at(beh['trial_number']).astype(int)
    remap = {int(o): int(n) for o, n in zip(orig, trials.number)}
    tr = np.array([remap.get(x, -1) for x in trn])
    good = (lik > 0.5) & np.isfinite(rad) & (tr > 0)
    return dict(t=t, X=X, Y=Y, cx=cx, cy=cy, rad=rad, pos=pos, spd=spd, tr=tr,
                good=good, trials=trials)


def population(mo, dy):
    """The canonical per-cell labels, ordered by each cell's PC1 loading."""
    z = load_session_labels(mo, dy)
    if z is None:
        raise RuntimeError(f'no anchoring labels for M{mo}D{dy}')
    keep = np.isfinite(z['pc1_load'])
    # Cells that never leave one mode are split off to the right of a gap, as in
    # fig1_v2. Their PC1 loading is ~0 by construction -- a constant row
    # contributes nothing to the decomposition -- so ordering purely by loading
    # buries them mid-raster, where they read as weak followers rather than as
    # cells that never varied at all.
    _L = z['labels'][keep]
    _load = z['pc1_load'][keep]
    _varies = np.nanstd(_L, axis=1) > 0
    _ov = np.where(_varies)[0][np.argsort(_load[_varies])[::-1]]
    _lk = np.where(~_varies)[0]
    _lk = _lk[np.argsort(-np.array([np.nanmean(_L[i]) for i in _lk]))] if len(_lk) else _lk
    order = np.concatenate([_ov, _lk]) if len(_lk) else _ov
    # major anchoring transitions, marked on every panel that carries the trial
    # axis so a block can be followed from the raster into the pupil rows
    st = median_filter((z['frac_anch'] > .5).astype(float), size=9,
                       mode='nearest') > .5
    tb = [b - .5 for b in np.where(np.diff(st.astype(int)) != 0)[0] + 1]
    return dict(L=z['labels'][keep][order], n_varies=int(len(_ov)),
                n_locked=int(len(_lk)), trial=z['trial'].astype(int),
                frac=z['frac_anch'], pc1=z['pc1'], trans=tb,
                var=float(z['pc1_var']), n=int(keep.sum()))


def state_blocks(mo, dy, S, P):
    """Longest anchored and non-anchored block, from the median-filtered state."""
    st = median_filter((P['frac'] > 0.5).astype(float), size=9,
                       mode='nearest') > 0.5
    tn = P['trial']
    runs, s0 = [], 0
    for i in range(1, len(st)):
        if st[i] != st[i - 1]:
            runs.append((tn[s0], tn[i - 1], st[s0], i - s0)); s0 = i
    runs.append((tn[s0], tn[-1], st[s0], len(st) - s0))
    best = {}
    for want in (True, False):
        cand = [r for r in runs if r[2] == want]
        best['anch' if want else 'non'] = max(cand, key=lambda r: r[3]) if cand else None
    return best


def pick_trial(S, a, b):
    """The trial in [a,b] whose position profile is closest to the block mean."""
    prof, keep = {}, []
    for u in range(a, b + 1):
        vals = []
        for target in SHOW_POS:
            m = (S['tr'] == u) & S['good'] & (S['spd'] >= 3.0) & \
                (np.abs(S['pos'] - target) <= POS_TOL)
            vals.append(S['rad'][m].mean() if m.sum() else np.nan)
        if not np.isnan(vals).any():
            prof[u] = np.array(vals); keep.append(u)
    if not keep:
        return None, None
    mean_prof = np.nanmean(np.array([prof[u] for u in keep]), axis=0)
    best = min(keep, key=lambda u: np.sqrt(np.mean((prof[u] - mean_prof) ** 2)))
    return best, prof[best]


def trial_frames(S, u):
    out = []
    for target in SHOW_POS:
        m = np.where((S['tr'] == u) & S['good'] & (S['spd'] >= 3.0) &
                     (np.abs(S['pos'] - target) <= POS_TOL))[0]
        out.append(int(m[np.argmin(np.abs(S['pos'][m] - target))]) if len(m) else None)
    return out


def build(mo, dy):
    S = load_session(mo, dy)
    P = population(mo, dy)
    blocks = state_blocks(mo, dy, S, P)
    CACHE = f'{FIG}/eye_frames_M{mo}D{dy}_trials.npz'
    vid = find_video(mo, dy)
    if vid is None and not os.path.exists(CACHE):
        raise RuntimeError(f'no video for M{mo}D{dy} and no cached frames')
    zone = (mo, dy) in EYE_ZONE_OFFSET if vid is None else \
        'eye_zone' in os.path.basename(vid)
    if zone:
        ox, oy = EYE_ZONE_OFFSET[(mo, dy)]
        cw = ch = None
    else:
        ox, oy = 0.0, 0.0
        cx0 = cy0 = cw = ch = None
        if not os.path.exists(CACHE):
            C = crops()
            r = C[(C.mouse == mo) & (C.day == dy) &
                  (C.session.astype(str).str.lower() == 'vr')].iloc[0]
            cx0, cy0, cw, ch = int(r.x), int(r.y), int(r.w), int(r.h)
    print(f'M{mo}D{dy}: {os.path.basename(vid) if vid else "cached frames"} '
          f'({"pre-cropped" if zone else "full frame"})')

    missing = [lab for k, lab in (('anch', 'anchored'), ('non', 'non-anchored'))
               if blocks[k] is None]
    if missing:
        raise RuntimeError(
            f'M{mo}D{dy} has no {" or ".join(missing)} trials under the gated '
            f'classifier (frac_anch {np.nanmean(P["frac"]):.2f} across '
            f'{len(P["trial"])} trials), so it cannot be a two-state example')

    sel = {}
    for k, lab in (('anch', 'anchored'), ('non', 'non-anchored')):
        a, b, _, n = blocks[k]
        u, prof = pick_trial(S, a, b)
        if u is None:
            raise RuntimeError(f'no complete trial in the {lab} block {a}-{b}')
        sel[k] = dict(block=(a, b, n), trial=u, prof=prof, frames=trial_frames(S, u))
        print(f'  {lab:13s} block trials {a}-{b} ({n}) -> trial {u}, '
              f'radii {np.round(prof,1)}')

    want = [f for k in sel for f in sel[k]['frames'] if f is not None]
    store = {}
    if os.path.exists(CACHE):
        z = np.load(CACHE); store = {int(k): z[k] for k in z.files}
    todo = [i for i in want if i not in store]
    if todo:
        if vid is None:
            raise RuntimeError(f'{len(todo)} frames missing from the cache and no '
                               'video available -- mount /Volumes/INCR-NolanLab')
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            raise RuntimeError(_sm_resolve_path(f'cannot open {vid} -- is @SM_VOLUMES_INCR_NOLANLAB_ROOT@ mounted?'))
        t0 = time.time()
        for i in todo:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
            if not ok:
                continue
            store[i] = fr if zone else fr[cy0:cy0 + ch, cx0:cx0 + cw]
        cap.release()
        print(f'  read {len(todo)} frames in {time.time()-t0:.0f}s')
        if store:
            np.savez_compressed(CACHE, **{str(k): v for k, v in store.items()})
    miss = [i for i in want if i not in store]
    if miss:
        raise RuntimeError(f'{len(miss)} frames unreadable; refusing to draw blanks')

    # ---- figure ----
    NB_MEAN = 25                 # position bins for the mean curve (8 cm)
    NB_MAP = 40                  # finer bins for the per-trial heatmap (5 cm)
    ntr = len(P['trial'])
    tick_i = [i for i, t in enumerate(P['trial']) if t % 20 == 0]
    tick_l = [int(P['trial'][i]) for i in tick_i]
    fig = plt.figure(figsize=(13.4, 5.6))
    outer = fig.add_gridspec(2, 3, width_ratios=[1.00, 2.5, 1.15],
                             height_ratios=[.62, .84], wspace=.24, hspace=.50)
    # the apparatus drawings side by side, pupillometry to the LEFT
    gap = outer[0, 0].subgridspec(1, 2, wspace=.10)
    # row 2 runs the full width so the five panels sit on one line. Giving the
    # colorbar its own full gridspec column at the SAME wspace as every other
    # boundary (.46) put it nowhere near the heatmap it belongs to -- a wide gap
    # on the left of the colorbar and nothing distinguishing "G's colorbar" from
    # "a sixth panel". It also squeezed H, whose paired-dot comparison needs
    # real width, and left H's rotated y-label sitting close enough to the
    # colorbar's own label to overlap.
    #
    # Fixed by nesting: G and its colorbar are ONE gridspec cell (GC), split
    # internally with a TIGHT wspace so the bar hugs the heatmap, while GC as a
    # whole still gets the full .46 wspace against its neighbours on both sides,
    # same as every other panel boundary.
    #
    # The 5th column is a near-zero placeholder, not a real panel width. I's
    # axes gets created here only to get a y-extent; its x0 and width are
    # thrown away a few lines below, where I is re-positioned to match D's
    # x-span exactly (see "D and I share an x-axis" below).
    #
    # H's width ALSO ends up overridden below rather than coming from its
    # ratio here (1.15 is just a reasonable starting box for it to sit in
    # before that override). Tried shrinking the placeholder column to
    # "reclaim" that space for H through the ratio math directly first, but
    # grow's columns are renormalised by the ratio SUM, so shrinking one
    # column rescales every column's absolute width -- including the ones
    # before it (E, F, G) that have nothing to do with H or I. Pinning H's
    # right edge explicitly, once I's real (D-aligned) left edge is known,
    # decouples the two instead of fighting that renormalisation.
    grow = outer[1, :].subgridspec(1, 5,
                                   width_ratios=[1.50, .40, 1.32, 1.15, .05],
                                   wspace=.46)
    ggc = grow[2].subgridspec(1, 2, width_ratios=[1.22, .085], wspace=.07)
    # Left column: the two apparatus drawings. These are vector PDFs, which
    # matplotlib cannot embed, so the axes are left EMPTY here and the artwork is
    # placed into the saved PDF afterwards by _place_apparatus(). Rasterising
    # them into imshow would have been simpler and would have thrown away the
    # line work at exactly the sizes where it matters.
    apparatus = []
    for col, srcname in ((0, 'camera_and_IR_mouse.pdf'),
                         (1, 'pupilometry.pdf')):
        axp_ = fig.add_subplot(gap[col])
        axp_.set_xticks([]); axp_.set_yticks([]); axp_.axis('off')
        apparatus.append((axp_, srcname))
    # the eye strips
    gl = outer[0, 1].subgridspec(2, len(SHOW_POS), hspace=.12, wspace=.06)
    # Raster and PC1, trials running DOWN so they can be read across into the
    # pupil heatmap. Widths are fig1_v2's: 1.38 to .38, a ratio of 3.6 to 1, so
    # the PC1 strip stays a strip. Holding that ratio inside the left column
    # would have stretched the raster, so the leading .45 is an empty spacer --
    # no axes is drawn there -- and the pair keeps roughly the width it had.
    gb = grow          # raster at grow[0], PC1 at grow[1]
    # D, E: mean pupil radius above the per-trial heatmap, sharing position
    gr = outer[0, 2].subgridspec(3, 1, height_ratios=[.10, 1, .10], hspace=0)

    MCX = np.median(S['cx'][S['good']]) + ox
    MCY = np.median(S['cy'][S['good']]) + oy
    for r, (k, col, lab) in enumerate((('anch', ANCH_COLOR, 'anchored'),
                                       ('non', NONANCH_COLOR, 'non-anchored'))):
        d = sel[k]
        for c, fi in enumerate(d['frames']):
            ax = fig.add_subplot(gl[r, c])
            ax.imshow(cv2.cvtColor(store[fi], cv2.COLOR_BGR2RGB))
            for jj, cc in enumerate(LM_COLORS):
                ax.scatter(S['X'][jj, fi] + ox, S['Y'][jj, fi] + oy, s=5.5,
                           color=cc, edgecolor='k', linewidth=.2, zorder=4)
            ax.set_xlim(MCX - ZOOM, MCX + ZOOM)
            ax.set_ylim(MCY + ZOOM * .78, MCY - ZOOM * .78)
            ax.text(.5, -.20, f'{S["rad"][fi]:.1f}', transform=ax.transAxes,
                    fontsize=6.5, color='0.25', ha='center')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(col); sp.set_linewidth(1.1)
            if r == 0:
                ax.set_title(f'{SHOW_POS[c]} cm', fontsize=7.5, color='0.25', pad=3)
            if c == 0:
                ax.set_ylabel(f'{lab}\ntrial {d["trial"]}', fontsize=8, color=col,
                              labelpad=6)

    def mark_trials(ax, x0):
        """Where the two displayed traversals sit on the trial axis.

        No state strip: PC1 already carries the anchored/non-anchored structure,
        and a third rendering of it in the margin says nothing the raster and
        PC1 have not said. What is NOT in either of them is which single trial
        the eye frames came from, so that keeps a marker.
        """
        for k, col in (('anch', ANCH_COLOR), ('non', NONANCH_COLOR)):
            iu = int(np.where(P['trial'] == sel[k]['trial'])[0][0])
            ax.plot([x0], [iu], transform=ax.get_yaxis_transform(),
                    clip_on=False, marker='<', ms=4.5, color=col, zorder=6)

    # B: the raster
    ax = fig.add_subplot(grow[0])
    TA_CMAP = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
    TA_NORM = BoundaryNorm([-.5, .5, 1.5], TA_CMAP.N)
    # NaN columns render as figure background and separate the two groups
    _nv, _nl = P['n_varies'], P['n_locked']
    if _nl:
        _gap = max(2, int(round(.025 * P['n'])))
        _Lr = np.vstack([P['L'][:_nv],
                         np.full((_gap, P['L'].shape[1]), np.nan),
                         P['L'][_nv:]])
    else:
        _gap, _Lr = 0, P['L']
    ax.imshow(_Lr.T, aspect='auto', cmap=TA_CMAP, norm=TA_NORM,
              interpolation='nearest', extent=[0, _Lr.shape[0], ntr - .5, -.5])
    for tb in P['trans']:
        ax.axhline(tb, color='0.15', lw=.9, ls='--', zorder=4)
    if _nl:
        ax.annotate('varies (PC1 order)', (_nv / 2, 1.012),
                    xycoords=('data', 'axes fraction'), ha='center',
                    fontsize=6.2, color='0.3')
        ax.annotate(f'locked ({_nl})', (_Lr.shape[0] - _nl / 2, 1.012),
                    xycoords=('data', 'axes fraction'), ha='center',
                    fontsize=6.2, color='0.3')
    ax.set_xlabel(f'Cell (n={P["n"]})', fontsize=8.5)
    ax.set_ylabel('Trial', fontsize=8.5)
    ax.set_yticks(tick_i); ax.set_yticklabels(tick_l)
    ax.tick_params(labelsize=7.5)
    # pad 12, not 4: the varies/locked group labels sit at y = 1.012 in axes
    # fraction and print through a title set tight against the axes
    ax.set_title('MEC population anchoring', fontsize=9, pad=12)
    for sp in ax.spines.values():
        sp.set_visible(False)
    mark_trials(ax, -.022)
    axE = ax

    # C: PC1, the same raster as one number per trial
    ax = fig.add_subplot(grow[1])
    axF = ax
    yy = np.arange(ntr)
    pc = np.nan_to_num(P['pc1'])
    ax.fill_betweenx(yy, 0, pc, where=pc >= 0, color=ANCH_COLOR, linewidth=0,
                     edgecolor='none', interpolate=True)
    ax.fill_betweenx(yy, 0, pc, where=pc < 0, color=NONANCH_COLOR, linewidth=0,
                     edgecolor='none', interpolate=True)
    ax.axvline(0, color='0.4', lw=.7)
    for tb in P['trans']:
        ax.axhline(tb, color='0.15', lw=.9, ls='--', zorder=4)
    ax.set_ylim(ntr - .5, -.5)
    ax.set_xticks([])
    ax.tick_params(labelleft=False, left=False)
    ax.set_title(f'PC1\n({100 * P["var"]:.0f}% var)', fontsize=8, pad=4)
    # no spines: the zero line is the only meaningful reference, and a left
    # spine would sit at the most negative score rather than at zero
    for sp in ax.spines.values():
        sp.set_visible(False)

    # D: mean pupil radius against position, one curve per state block
    ax = fig.add_subplot(gr[1])
    x = np.arange(NB_MEAN) * (200.0 / NB_MEAN) + 4
    for k, col, lab in (('anch', ANCH_COLOR, 'anchored'),
                        ('non', NONANCH_COLOR, 'non-anchored')):
        a, b, _ = sel[k]['block']
        m = S['good'] & (S['spd'] >= 3.0) & (S['tr'] >= a) & (S['tr'] <= b)
        pb = np.clip((S['pos'] / (200.0 / NB_MEAN)).astype(int), 0, NB_MEAN - 1)
        # SEM across TRIALS, not across frames. Frames within a trial are highly
        # correlated -- the pupil barely moves in 33 ms -- so treating ~350 frames
        # per bin as independent understates the error by 2-3x (0.03 px instead of
        # 0.11 px here). The trial is the independent unit.
        us = np.unique(S['tr'][m])
        mu, se = np.zeros(NB_MEAN), np.zeros(NB_MEAN)
        for i in range(NB_MEAN):
            bm = m & (pb == i)
            pt = np.array([S['rad'][bm & (S['tr'] == u)].mean() for u in us])
            pt = pt[np.isfinite(pt)]
            mu[i] = pt.mean() if len(pt) else np.nan
            se[i] = pt.std(ddof=1) / np.sqrt(len(pt)) if len(pt) > 1 else np.nan
        ax.fill_between(x, mu - se, mu + se, color=col, alpha=.35, linewidth=0,
                        edgecolor='none', zorder=2)
        ax.plot(x, mu, color=col, lw=1.7, zorder=3, label=f'{lab} block')
        # the displayed trial's whole traversal, binned at 1 cm
        u = sel[k]['trial']
        mt = S['good'] & (S['spd'] >= 3.0) & (S['tr'] == u)
        pb1 = np.clip(S['pos'].astype(int), 0, 199)
        tv = np.array([S['rad'][mt & (pb1 == i)].mean() if (mt & (pb1 == i)).any()
                       else np.nan for i in range(200)])
        ok1 = np.isfinite(tv)
        ax.plot(np.arange(200)[ok1], tv[ok1], '-', color=col, lw=.8, alpha=.75,
                zorder=4, label=f'trial {u}')
    ax.axvspan(90, 110, color='#d8e4d0', alpha=.55, lw=0, zorder=0)
    for x0 in (0, 170):
        ax.axvspan(x0, x0 + 30, color='0.90', lw=0, zorder=0)
    ax.set_xlim(0, 200); ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xticklabels([])
    ax.set_ylabel('Pupil radius (px)', fontsize=8.5)
    ax.legend(fontsize=6.2, frameon=False, loc='upper right', ncol=1)
    ax.tick_params(labelsize=7.5); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(f'M{mo} D{dy}', fontsize=9.5, loc='left', pad=6)
    axD = ax

    # E: the same radius, every trial, on the trial axis of B and C
    ax = fig.add_subplot(ggc[0])
    axG = ax
    H = np.full((ntr, NB_MAP), np.nan)
    pbm = np.clip((S['pos'] / (200.0 / NB_MAP)).astype(int), 0, NB_MAP - 1)
    ok = S['good'] & (S['spd'] >= 3.0)
    for i, u in enumerate(P['trial']):
        mu_ = ok & (S['tr'] == u)
        if not mu_.any():
            continue
        cnt = np.bincount(pbm[mu_], minlength=NB_MAP).astype(float)
        tot = np.bincount(pbm[mu_], weights=S['rad'][mu_], minlength=NB_MAP)
        H[i] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    cmap2 = plt.get_cmap('plasma').copy(); cmap2.set_bad('0.9')
    lo, hi = np.nanpercentile(H, [2, 98])
    im = ax.imshow(np.ma.masked_invalid(H), aspect='auto', cmap=cmap2,
                   vmin=lo, vmax=hi, interpolation='nearest',
                   extent=[0, 200, ntr - .5, -.5])
    for tb in P['trans']:
        ax.axhline(tb, color='0.15', lw=.9, ls='--', zorder=4)
    ax.set_xlim(0, 200); ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel('Position on track (cm)', fontsize=8.5)
    ax.set_yticks(tick_i); ax.set_yticklabels([])
    ax.tick_params(labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    mark_trials(ax, -.022)
    # The colorbar lives in ggc[1], a cell nested tightly against G (ggc[0]) so
    # it hugs the heatmap rather than floating in the middle of a full panel
    # gap; ggc as a WHOLE still gets the usual wspace against its neighbours.
    cax = fig.add_subplot(ggc[1])
    cb = fig.colorbar(im, cax=cax)
    # Title ABOVE the bar, not cb.set_label's default rotated label to its
    # RIGHT. That default label and H's own rotated y-label were both floating
    # outward from their axes boxes by a fixed point padding -- independent of
    # gridspec wspace -- and with the bar now close to G there was nowhere near
    # enough room between them; they overlapped. A title above the bar costs no
    # lateral space at all, so there is nothing left to collide with.
    cax.set_title('Pupil\nradius (px)', fontsize=6, pad=3, linespacing=1.1)
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)

    # ── bottom row: pupil dilation by state, and across state transitions ──
    # This row is the POPULATION result, not this session: every included
    # session contributes one value, so the example above is shown to be
    # representative rather than asserted to be.
    E = eye_trials()

    nsess = E.groupby(['mouse', 'day']).ngroups

    # dilation, anchored against non-anchored, paired within session
    axb = fig.add_subplot(grow[3])
    W = (E.groupby(['mouse', 'day', 'anch']).z_rad.mean().unstack()
         .rename(columns={True: 'a', False: 'n'}).dropna())
    for _, r in W.iterrows():
        axb.plot([0, 1], [r.a, r.n], color='0.78', lw=.6, zorder=1)
    for x_, v_, c_ in ((0, W.a, ANCH_COLOR), (1, W.n, NONANCH_COLOR)):
        axb.scatter(np.full(len(W), x_) + np.random.default_rng(0)
                    .uniform(-.06, .06, len(W)), v_, s=11, color=c_, lw=0,
                    alpha=.8, zorder=2)
        axb.errorbar(x_, v_.mean(), yerr=v_.std(ddof=1) / np.sqrt(len(W)),
                     color='k', marker='_', ms=15, lw=1.4, capsize=3, zorder=3)
    pb = wilcoxon(W.a, W.n).pvalue
    axb.axhline(0, color='0.75', lw=.8, zorder=0)
    axb.set_xlim(-.4, 1.4); axb.set_xticks([0, 1])
    axb.set_xticklabels(['anchored', 'non-\nanchored'], fontsize=7.5)
    axb.get_xticklabels()[0].set_color(ANCH_COLOR)
    axb.get_xticklabels()[1].set_color(NONANCH_COLOR)
    axb.set_ylabel('z(pupil radius)', fontsize=8.5)
    # stacked rather than run on one line: H is the narrowest panel in the row,
    # and titles are not clipped to the axes box, so a wide one overhangs into I
    axb.set_title(f'Δ = {(W.a - W.n).mean():+.3f}\np = {pb:.1g}\n'
                  f'({len(W)} sessions)', fontsize=7.5, loc='left', color='0.25')
    axb.tick_params(labelsize=7.5); axb.spines[['top', 'right']].set_visible(False)

    # dilation against track position, so the difference can be localised
    axpp = fig.add_subplot(grow[4])
    A, N, K = position_profiles(E)
    xpos = (np.arange(NPOS) + .5) * (200.0 / NPOS)
    for M_, col_, lab_ in ((A, ANCH_COLOR, 'anchored'),
                           (N, NONANCH_COLOR, 'non-anchored')):
        mu = np.nanmean(M_, axis=0)
        se = np.nanstd(M_, axis=0) / np.sqrt(np.sum(np.isfinite(M_), axis=0))
        axpp.fill_between(xpos, mu - se, mu + se, color=col_, alpha=.30,
                          linewidth=0, edgecolor='none', zorder=2)
        axpp.plot(xpos, mu, color=col_, lw=1.6, zorder=3, label=lab_)
    # where does the difference actually live? Paired across sessions, bin by
    # bin. Uncorrected -- 50 bins are not independent, neighbouring bins share
    # the same traversals -- so this marks extent, it does not test it.
    pv = np.array([wilcoxon(A[:, i], N[:, i]).pvalue
                   if np.isfinite(A[:, i] - N[:, i]).all() else np.nan
                   for i in range(NPOS)])
    ytop = axpp.get_ylim()[1]
    sig = pv < .05
    axpp.scatter(xpos[sig], np.full(sig.sum(), ytop * .96), s=3, marker='s',
                 color='0.35', zorder=4)
    axpp.axvspan(90, 110, color='#d8e4d0', alpha=.55, lw=0, zorder=0)
    for x0_ in (0, 170):
        axpp.axvspan(x0_, x0_ + 30, color='0.90', lw=0, zorder=0)
    axpp.axhline(0, color='0.75', lw=.8, zorder=1)
    axpp.set_xlim(0, 200); axpp.set_xticks([0, 50, 100, 150, 200])
    axpp.set_xlabel('Position on track (cm)', fontsize=8.5)
    axpp.set_ylabel('z(pupil radius)', fontsize=8.5)
    # the point of this panel is that the shift is TONIC: present at every
    # position rather than concentrated at the reward zone or any other landmark,
    # which is what separates a change in engagement from attention paid at
    # selective points along the track
    _same = int((np.nanmean(A - N, axis=0) < 0).sum())
    axpp.set_title(f'lower across the WHOLE track ({len(K)} sessions)\n'
                   f'{_same}/{NPOS} bins same sign, {int(sig.sum())}/{NPOS} p < 0.05',
                   fontsize=7.5,
                   loc='left', color='0.25')
    axpp.legend(fontsize=6.5, frameon=False, loc='lower right')
    axpp.tick_params(labelsize=7.5)
    axpp.spines[['top', 'right']].set_visible(False)

    print(f'  bottom row: {nsess} sessions, dilation Δ = {(W.a - W.n).mean():+.3f} '
          f'(p = {pb:.2g})')

    out = f'{FIG}/fig5_pupil_arousal_M{mo}D{dy}.pdf'
    # ---- panel letters -----------------------------------------------------
    # Placed from the GRIDSPEC cells rather than from individual axes: several
    # panels are groups (the eye strips are a 2 x 7 subgrid, the raster and its
    # PC1 strip sit inside a 3-column subgrid), and lettering the first axes of a
    # group puts the letter in the middle of the block it is meant to label.
    def _panel(cell, letter, dx=-.018, dy=.012):
        bb = cell.get_position(fig)
        fig.text(bb.x0 + dx, bb.y1 + dy, letter, fontsize=10, weight='bold',
                 va='bottom', ha='left')

    # D and I both plot z(pupil radius) against the SAME track-position axis, one
    # above the other on the page, but they come from different parent gridspecs
    # (D from the 3-column top-level row, I from row 2's 6-column strip) and so
    # never shared an x-span by construction. Forcing I's left edge and width to
    # match D's exactly makes the two genuinely comparable at a glance -- a
    # feature directly below D sits at the same track position in I.
    pD = axD.get_position()
    pI = axpp.get_position()
    axpp.set_position([pD.x0, pI.y0, pD.width, pI.height])

    # Now make room for H. Measured, row 2 had NEGATIVE space for it: the left
    # block (E, F, G, colorbar) ended at .642 and I -- pinned to D just above --
    # starts at .740, leaving .098 for two .052 panel gaps plus H itself, i.e.
    # -.006. No set of grow width_ratios fixes that, because I's left edge comes
    # from D's gridspec and does not move when row 2's ratios change; the ratios
    # only decide how row 2 divides space it has already run out of. That is why
    # widening H's own column kept pushing it straight into I.
    #
    # The space has to come from the left block, so it is taken explicitly:
    # every left-block axes is scaled horizontally about E's left edge by one
    # common factor, which preserves their relative widths and the tight
    # heatmap-to-colorbar spacing exactly. The raster and heatmap lose ~20% of
    # their width; both are wide panels showing trial x position structure and
    # still read at that size, whereas H could not be read at all before.
    # The gaps are solved against INK, not against the axes boxes. Every one of
    # these panels floats labels outside its own box by a fixed point padding
    # that no gridspec setting accounts for -- the colorbar's tick numbers to its
    # right, H's y-label and tick numbers to its left, I's y-label to its left --
    # so boxes that are a comfortable distance apart can still have their text
    # overlapping. The overhangs are measured once from the drawn figure and the
    # boxes placed so the INK clears, which is what actually has to be true.
    H_WIDTH = .105
    CLEAR = .014            # whitespace between neighbouring panels' ink
    left_block = [axE, axF, axG, cax]
    fig.canvas.draw()
    _inv = fig.transFigure.inverted()

    def _ink(a):
        return a.get_tightbbox(fig.canvas.get_renderer()).transformed(_inv)

    lb_x0 = min(a.get_position().x0 for a in left_block)
    lb_x1 = max(a.get_position().x1 for a in left_block)
    ov_lb = max(_ink(a).x1 for a in left_block) - lb_x1   # colorbar numbers
    ov_hl = axb.get_position().x0 - _ink(axb).x0          # H's y-label + ticks
    ov_hr = _ink(axb).x1 - axb.get_position().x1          # H's right tick label
    ov_il = pD.x0 - _ink(axpp).x0                         # I's y-label + ticks

    # H sits as far right as it can while its ink clears I's, then the left
    # block is scaled about its own left edge until its ink clears H's.
    h_x1 = pD.x0 - ov_il - CLEAR - ov_hr
    h_x0 = h_x1 - H_WIDTH
    lb_target = h_x0 - ov_hl - CLEAR - ov_lb
    s = (lb_target - lb_x0) / (lb_x1 - lb_x0)
    for a in left_block:
        p = a.get_position()
        a.set_position([lb_x0 + (p.x0 - lb_x0) * s, p.y0, p.width * s, p.height])
    pH = axb.get_position()
    axb.set_position([h_x0, pH.y0, H_WIDTH, pH.height])

    # Only the top row is lettered from its gridspec cells. Every panel in row 2
    # has had its position overridden above, and a SubplotSpec still reports the
    # cell it was allotted rather than where its axes ended up, so lettering row
    # 2 from the cells strands each letter at the panel's old location.
    for _cell, _ltr in ((gap[0], 'A'),            # camera and IR schematic
                        (gap[1], 'B'),            # pupillometry drawing
                        (outer[0, 1], 'C'),       # eye frames along the track
                        (outer[0, 2], 'D')):      # mean radius by position
        _panel(_cell, _ltr)
    for _ax, _ltr in ((axE, 'E'),                 # anchoring raster
                      (axF, 'F'),                 # PC1
                      (axG, 'G'),                 # per-trial radius heatmap
                      (axb, 'H'),                 # anchored vs non, paired
                      (axpp, 'I')):               # dilation by position
        _p = _ax.get_position()
        fig.text(_p.x0 - .018, _p.y1 + .012, _ltr, fontsize=10,
                 weight='bold', va='bottom', ha='left')

    # tight bbox crops the page, so axes positions must be expressed relative to
    # the crop before they mean anything in PDF points
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    boxes = [(ax_.get_position(), nm) for ax_, nm in apparatus]
    plt.savefig(out, dpi=220, bbox_inches='tight'); plt.close(fig)
    _place_apparatus(out, boxes, tb, fig.get_size_inches())
    print('  saved', out)


if __name__ == '__main__':
    # M28 D18 was dropped: it has 0 ENTm spatial cells (all visual cortex), so its
    # anchoring state was not a MEC state at all.
    for mo, dy in ((21, 19), (29, 17)):
        try:
            build(mo, dy)
        except RuntimeError as e:
            print(f'  ! skipped: {e}')

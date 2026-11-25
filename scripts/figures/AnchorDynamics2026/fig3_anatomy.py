"""Figure 3 — where MEC spatial cells sit, and what position predicts.

Usage:  python3 fig3_anatomy.py

    A     the gradient shown rather than summarised: open-field rate maps from
          four M25 sessions whose probes sat at different mediolateral
          positions, sampled PROPORTIONALLY so the grid:non-grid ratio drawn is
          the ratio actually recorded
    B, C  best-fit slices through the recorded cells -- one mouse (M25) and all
          mice pooled -- each under a small 3D view of the brain showing where
          the probe sat. Cells coloured by open-field identity over the Allen
          region annotation.
    D-F   identity across the anatomical axes: the grid fraction against
          mediolateral and dorsoventral position, and superficial against deep.
          Anterior-posterior is shown as superficial vs deep rather than as a
          coordinate, because in a structure as curved as MEC the AP coordinate
          is mostly a proxy for layer, and layer is the quantity with a
          mechanism attached to it.
    G-I   the same axes against whether a cell FOLLOWS the population anchoring
          state, pooled and within session.

SCOPE: THIS FIGURE IS ABOUT ANATOMY WITHIN MEC. The coarser question -- whether
a cell has to be in MEC at all, scored against each cell's own circular-shift
null -- is a between-region comparison and lives with the other regional panels
in the Figure 1 supplement (fig1_supp_nonmec.py).

ONE AXIS DOES EVERYTHING. Mediolateral position predicts both what a cell is and
whether it follows; dorsoventral position and probe depth predict neither; layer
predicts identity only.

    axis      predicts            within session (correct restriction)
    M-L       grid identity       rho -0.078, 18/27 sessions, p = 0.025
    M-L       following           rho -0.109, 19/27 sessions, p = 0.034
    layer     grid identity       8.9% sup vs 3.9% deep, 21/31, p = 0.002
    layer     following           rho +0.028, 25/39 sessions, p = 0.154  (NOT a panel)
    D-V       grid identity       p = 0.90
    D-V       following           p = 0.99
    A-P       grid identity       p = 0.10
    depth     following           p = 0.92

LAYER AGAINST FOLLOWING IS NOT PLOTTED, AND THE ROW ABOVE IS WHY IT IS STILL
RECORDED HERE. The panels test M-L, D-V and probe DEPTH against following; depth
is not layer, so for a while nothing in this file tested the quantity the text
was claiming when it called the responsive subnetwork "superficial". It does not
hold up at the standard this figure applies everywhere else: pooled, superficial
cells agree with the population axis more than deep ones (+0.162 vs +0.134,
p = 0.004), but pooling is unsafe here for exactly the reason it is unsafe
mediolaterally, and within session the effect is a non-significant trend
(+0.158 vs +0.139 paired, 24/36 sessions, p = 0.077). The superficial bias in
following is therefore the one carried by IDENTITY -- layer predicts what a cell
is, and what a cell is predicts following -- not a gradient in following of its
own. Run the test again before anyone restores the word to a heading.

WHICH SESSIONS CAN ANSWER THE QUESTION. Mediolateral position is assigned PER
SHANK, so a single-shank session holds no mediolateral variation at all and
contributes only noise -- 33 of 61 sessions span under 50 um. The within-session
test therefore runs on the 27 sessions spanning at least 200 um, two shank
pitches or more.

This matters more than it sounds. An earlier version selected sessions by
`nunique > 3`, the count of distinct mediolateral values, which EXCLUDES the
cleanly sampled 2-, 3- and 4-shank sessions (a 3-shank session holds three
discrete values while spanning 510 um) and ADMITS single-shank sessions whose
coordinates differ only by sub-micron numerical jitter. It ran the test almost
entirely on sessions with no mediolateral variation and reported p = 1.00 for an
effect that is present at p = 0.034.

THE CONTROL THAT MAKES THE RESTRICTION HONEST: for dorsoventral position and
probe depth the criterion changes nothing (p = 0.99 and 0.92 either way),
because every session varies along the shank. Only the axis sampled by shanks is
affected, so the restriction is not manufacturing the effect.

WHAT CANNOT BE SETTLED HERE. Mediolateral position predicts identity, and
identity predicts following, so the mediolateral gradient in following may be
nothing more than the identity gradient seen through it. Restricting to non-grid
cells leaves the effect size almost unchanged (rho -0.100 against -0.109) but
loses significance at 26 sessions (p = 0.111). The two accounts are not
separable at this sample size and the figure does not claim otherwise.
"""
import math
import os
import re
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from tifffile import imread

plt.rcParams['font.family'] = 'Arial'
# panel letters are set as mathtext \bf inside the axes titles, so mathtext
# must resolve to Arial Bold rather than the default DejaVu
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'


def _lp(ax, s, dx=-.16, dy=1.0):
    """Bold panel letter, drawn separately from the title (see fig2_single_units.py
    for why: inline mathtext ties the letter's size to that panel's own title
    fontsize, which varies panel to panel)."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
RENDERS = f'{ROOT}/data/anatomy_renders'
BG = '/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2'
DC_PATH = '/Users/harryclark/Documents/spatial-manifolds/data/device_contact_id_annotations.csv'  # build_device_contacts.py, from clark2025

# Region shading. The previous palette gave MEC a teal-green and visual cortex
# a purple, which read as the non-anchored teal and the anchored magenta on the
# rasters in H and I. These avoid both state colours, the red and blue that
# Figure 2 reserves for cell types, the interneuron orange and the contact
# orange. MEC is split into superficial (L1-3) and deep (L5-6), the paper's own
# layer split; there are no layer-4 cells in the data.
COL_HIPPO, COL_PREPARA = '#BDBDBD', '#A3A85E'
COL_MEC_SUP, COL_MEC_DEEP = '#E3BC4F', '#8C5A2B'
COL_VIS, COL_REGION_OTHER = '#7F8C99', '#E8E8E8'
COL_MEC = COL_MEC_SUP                     # kept for any older reference
# GREY KEY, as Figure S7: the cells carry the colour, the tissue only says where
# it is. One grey per class, used for the slices, the raster strips, the legend
# and the 3D renders alike. S7's own levels put visual cortex and pre/para at
# 0.90 and 0.87, indistinguishable on a strip, so these are spread a little
# further; deep MEC is a step darker than superficial to keep the layer split.
REGION_GREY = {1: .96, 2: .86, 3: .80, 4: .69, 5: .91}     # by region class
GREY_OTHER = .975
REGION_ALPHA = .55
REGION_COLS = (COL_HIPPO, COL_PREPARA, COL_MEC_SUP, COL_MEC_DEEP, COL_VIS)
REGION_NAMES = ('Hippocampus', 'Pre/parasubiculum', 'MEC superficial (L1–3)',
                'MEC deep (L5–6)', 'Visual cortex')


def _blend(hexcol, a=REGION_ALPHA):
    """The colour a region patch shows on white paper at alpha a."""
    rgb = np.array([int(hexcol.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return tuple(a * rgb + (1 - a))
COL_GC, COL_NGS, COL_OTHER = '#c04744', '#3171ae', '#888888'
COL_CONTACT = '#ff6500'
CELL_ALPHA = .75
ML_JITTER_STD = 40.0
ML_JITTER_FIG4 = 15.0         # um, display jitter on the Figure 4 slices
# a session must span two shank pitches before a within-session correlation
# along the mediolateral axis means anything -- see the module docstring
MIN_SPAN_UM = 200.0
# ML_BOUNDARY removed 2026-10-03. The 3400 um medial/lateral split was drawn in
# the slice and binned panels but no criterion for it was ever stated, and every
# within-session test in this figure runs on the CONTINUOUS coordinate, so the
# line asserted a threshold the analysis does not use.

# ── CCF transform and region shading, as in grid_cell_anatomy_v2 ─────────────
annotations_set = imread(f'{BG}/annotation.tiff')
structure_set = pd.read_csv(f'{BG}/structures.csv')
id_to_acr = dict(zip(structure_set['id'], structure_set['acronym']))


def stereo_to_ccf(SC, angle=-0.0873):
    SC = np.asarray(SC, float)
    stretch = SC / np.array([1.0, 0.9434, 1.0])
    rotate = np.array([stretch[0] * math.cos(angle) - stretch[1] * math.sin(angle),
                       stretch[0] * math.sin(angle) + stretch[1] * math.cos(angle),
                       stretch[2]])
    return rotate + np.array([5400.0, 440.0, 5700.0])


def _df_to_ccf(d):
    return np.array([stereo_to_ccf(r) for r in d[['SC_z', 'SC_y', 'SC_x']].values])


def _dc_to_ccf(d):
    return np.array([stereo_to_ccf(r) for r in
                     d[['coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x']].values])


_RE = {1: re.compile(r'^(CA[1-4]?|DG)(-|$)'), 2: re.compile(r'^(PRE|PAR)\d*$'),
       3: re.compile(r'^ENTm[123]'), 4: re.compile(r'^ENTm[456]'),
       5: re.compile(r'^VIS')}


def region_class(acr):
    """0 other, 1 hippocampus, 2 pre/para, 3 MEC sup, 4 MEC deep, 5 visual."""
    for k, rx in _RE.items():
        if rx.match(str(acr)):
            return k
    return 0


def _ids_to_numeric(aid_flat):
    numeric = np.zeros(len(aid_flat), dtype=int)
    for sid in np.unique(aid_flat):
        if sid == 0:
            continue
        acr = id_to_acr.get(sid, 'root')
        for k, rx in _RE.items():
            if rx.match(acr):
                numeric[aid_flat == sid] = k
                break
    return numeric


cmap_list = [(1, 1, 1, 0)] + [(REGION_GREY[k],) * 3 + (1.0,) for k in (1, 2, 3, 4, 5)]
region_cmap = ListedColormap(cmap_list)
region_norm = BoundaryNorm(np.arange(len(cmap_list) + 1) - .5, region_cmap.N)

_df_regions = pd.read_csv(f'{ROOT}/data/cell_classifications_no_regions.csv')
_df_regions = _df_regions[_df_regions['mouse'] != 22]
_df_regions['SC_x'] = -np.abs(_df_regions['SC_x'])
_jit = np.random.default_rng(1)
_dc_all = pd.read_csv(DC_PATH)
_dc_all = _dc_all[_dc_all['mouse'] != 22]


def _apply_ml_jitter(d):
    d = d.copy()
    if ML_JITTER_STD > 0:
        d['SC_x'] = d['SC_x'] + _jit.normal(0, ML_JITTER_STD, size=len(d))
    return d


def get_mouse_pts(mouse=None):
    d = _df_regions if mouse is None else _df_regions[_df_regions['mouse'] == mouse]
    out = []
    for sel in (d['cell_type'] == 'GC', d['cell_type'] == 'NG',
                ~d['cell_type'].isin(['GC', 'NG'])):
        k = _apply_ml_jitter(d[sel].dropna(subset=['SC_x', 'SC_y', 'SC_z']))
        out.append(_df_to_ccf(k) if len(k) else np.empty((0, 3)))
    return out


def get_mouse_contacts(mouse=None):
    d = _dc_all if mouse is None else _dc_all[_dc_all['mouse'] == mouse]
    return _dc_to_ccf(d)


def fit_best_slice(fit_pts, extent_pts=None):
    """AP-DV-tilt-constrained best fit; the ML axis stays fixed in-plane.

    Fit to DEVICE CONTACTS, not to cell locations: per-cell ML is a per-shank
    nominal plus synthetic jitter, so fitting a plane to it would be fitting to
    the approximation rather than to the probe geometry that produced it.
    """
    if extent_pts is None:
        extent_pts = fit_pts
    centroid = fit_pts.mean(axis=0)
    ay_c = fit_pts[:, [0, 1]] - fit_pts[:, [0, 1]].mean(axis=0)
    _, _, Vt = np.linalg.svd(ay_c, full_matrices=False)
    major = Vt[0] if Vt[0][1] >= 0 else -Vt[0]
    minor = Vt[1] if Vt[1][0] >= 0 else -Vt[1]
    basis2 = np.array([major[0], major[1], 0.0])

    def project(pts):
        return 5700.0 - pts[..., 2], (pts - centroid) @ basis2

    def reconstruct(u, v):
        p = np.empty(np.broadcast(np.asarray(u), np.asarray(v)).shape + (3,))
        p[..., 0] = centroid[0] + v * basis2[0]
        p[..., 1] = centroid[1] + v * basis2[1]
        p[..., 2] = 5700.0 - u
        return p

    u_e, v_e = project(extent_pts)
    pu, pv = (u_e.max() - u_e.min()) * .08, (v_e.max() - v_e.min()) * .08
    return dict(u_lim=(u_e.min() - pu, u_e.max() + pu),
                v_lim=(v_e.min() - pv, v_e.max() + pv),
                tilt_deg=np.degrees(np.arctan2(minor[1], minor[0])),
                project=project, reconstruct=reconstruct)


def sample_region_grid(fit, step=10):
    u_vals = np.arange(*fit['u_lim'], step)
    v_vals = np.arange(*fit['v_lim'], step)
    V, U = np.meshgrid(v_vals, u_vals, indexing='ij')
    g = fit['reconstruct'](U, V)
    idx = [np.round(g[..., k] / 10).astype(int) for k in range(3)]
    sh = annotations_set.shape
    ok = np.ones(idx[0].shape, bool)
    for k in range(3):
        ok &= (idx[k] >= 0) & (idx[k] < sh[k])
    aid = np.zeros(idx[0].shape, int)
    aid[ok] = annotations_set[idx[0][ok], idx[1][ok], idx[2][ok]]
    return u_vals, v_vals, _ids_to_numeric(aid.ravel()).reshape(idx[0].shape)


def render_brain(mouse, path):
    """Small 3D view of the brain with MEC and this mouse's contacts.

    Rendered in a SUBPROCESS: brainrender opens a VTK window and loading the
    10 um atlas leaves state behind that interferes with matplotlib's Agg
    backend in the same interpreter. Cached, because the render takes minutes.
    """
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    code = f'''
import numpy as np, pandas as pd, math
from brainrender import Scene, settings as s
from brainrender.actors import Points
s.SHOW_AXES = False; s.BACKGROUND_COLOR = [1, 1, 1]; s.OFFSCREEN = True
# 'cartoon', the default, draws a silhouette line on every internal mesh edge,
# which at thumbnail size reads as scribble over the brain
s.SHADER_STYLE = 'plastic'
s.ROOT_ALPHA = 0.10
def t(SC, a=-0.0873):
    SC = np.asarray(SC, float); st = SC / np.array([1.0, 0.9434, 1.0])
    return np.array([st[0]*math.cos(a)-st[1]*math.sin(a),
                     st[0]*math.sin(a)+st[1]*math.cos(a), st[2]]) + np.array([5400.,440.,5700.])
dc = pd.read_csv({DC_PATH!r}); dc = dc[dc.mouse != 22]
{'dc = dc[dc.mouse == %d]' % mouse if mouse is not None else ''}
pts = np.array([t(r) for r in dc[['coord_SCs_z','coord_SCs_y','coord_SCs_x']].values])
sc = Scene(root=True, atlas_name='allen_mouse_10um', title='')
for nm, col, al in ([(f'ENTm{{k}}', {str(REGION_GREY[3])!r}, .6) for k in (1, 2, 3)]
                   + [(f'ENTm{{k}}', {str(REGION_GREY[4])!r}, .6) for k in (5, 6)]
                   + [('HPF', {str(REGION_GREY[1])!r}, .25)]):
    try: sc.add_brain_region(nm, alpha=al, color=col)
    except Exception: pass
sc.add(Points(pts, name='c', colors={COL_CONTACT!r}, radius=45, alpha=.9))
cen = pts.mean(0); F = (float(cen[0]), float(cen[1]), float(-cen[2]))
a = np.radians(320)
sc.render(camera={{'pos': (F[0]+30000*np.cos(a), F[1]-9000, F[2]-30000*np.sin(a)),
                  'focal_point': (6500., 4000., -5700.), 'viewup': (0,-1,0),
                  'clipping_range': (12000, 70000)}}, interactive=False, zoom=0.95)
sc.screenshot(name={path[:-4]!r}, scale=2)
sc.close()
'''
    # the first render loads the 10 um atlas and can take several minutes;
    # a timeout keeps a stalled VTK context from blocking the whole figure
    subprocess.run([sys.executable, '-c', code], check=True,
                   capture_output=True, timeout=1500)
    if not os.path.exists(path):
        raise RuntimeError('brainrender produced no screenshot')
    return path


def trim_white(img, pad=6):
    """Crop the white margin brainrender leaves around the brain."""
    a = img[..., :3] if img.ndim == 3 else img
    ink = (a.min(axis=2) < .97) if a.ndim == 3 else (a < .97)
    if not ink.any():
        return img
    ys, xs = np.where(ink)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


# ── quantitative data ───────────────────────────────────────────────────────
C = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
U = pd.read_csv(f'{PS}/unit_table.csv')[['mouse', 'day', 'cluster_id', 'r', 'identity']]
D = C.merge(U, on=['mouse', 'day', 'cluster_id'], how='left')
mec = D[D.brain_region.astype(str).str.startswith('ENTm')].copy()
mec['ml'] = mec.coord_SCs_x.abs()
mec['dv'] = mec.coord_SCs_y
mec['depth'] = mec.coord_probe_y
mec['layer'] = mec.brain_region.str.extract(r'ENTm(\d)')[0]
mec['sup'] = mec.layer.isin(['1', '2', '3'])
sp = mec[mec.cell_class_of1.isin(['GC', 'NGS'])].copy()
sp['gc'] = (sp.cell_class_of1 == 'GC').astype(float)
print(f'{len(mec)} MEC cells, {len(sp)} spatial, '
      f'{sp.groupby(["mouse", "day"]).ngroups} sessions, {sp.mouse.nunique()} mice')


def multi_shank(df, axis):
    span = df.groupby(['mouse', 'day'])[axis].agg(lambda v: v.max() - v.min())
    return set(span[span >= MIN_SPAN_UM].index)


def within_session(df, axis, yvar, restrict=True):
    sel = multi_shank(df, axis) if restrict else None
    ws = []
    for (mo, dy), g in df.dropna(subset=[yvar]).groupby(['mouse', 'day']):
        if sel is not None and (mo, dy) not in sel:
            continue
        if len(g) < 20 or g[axis].nunique() < 2 or g[yvar].nunique() < 2:
            continue
        r = spearmanr(g[axis], g[yvar])[0]
        if np.isfinite(r):
            ws.append(r)
    ws = np.array(ws)
    p = wilcoxon(ws).pvalue if len(ws) >= 6 else np.nan
    return ws, p


def binned(df, axis, yvar, nbin=8):
    q = df.dropna(subset=[axis, yvar])
    e = np.quantile(q[axis], np.linspace(0, 1, nbin + 1))
    e[-1] += 1e-6
    k = np.digitize(q[axis], e[1:-1])
    g = q.groupby(k)
    return (g[axis].mean().values, g[yvar].mean().values,
            (g[yvar].std(ddof=1) / np.sqrt(g[yvar].size())).values)



# ── M25 rate maps: the mediolateral gradient, shown rather than summarised ───
# Four M25 sessions whose probes sat at different mediolateral positions, from
# ~3000 um (medial) to ~3720 um (lateral). Cells are sampled PROPORTIONALLY --
# the grid:non-grid ratio drawn in each block is the ratio actually recorded in
# that session -- so the panel reports the gradient directly rather than
# illustrating it with hand-picked examples. Grid cells are taken in grid-score
# order and non-grid cells spread evenly across the dorsoventral extent so the
# block is not all one depth.
import pynapple as nap
from spatial_manifolds.detect_grids import (_fill_nans_from_neighbors,
                                            compute_travel_projected,
                                            gaussian_filter_nan)

SRC = '/Volumes/datasetsSD/Ephys_data/clark2025/'
RM_SESSIONS = [(24, 3000), (19, 3270), (22, 3540), (23, 3720)]
RM_N, RM_COLS = 27, 3          # 27 cells per session, 9 rows of 3
_cls = pd.read_csv(f'{ROOT}/data/cell_classifications.csv')


_OF_CACHE = {}


def _of_session(mouse, day):
    """Loaded once per session: 24 cells x 4 sessions is 96 maps, and reopening
    both NWB files for each of them dominates the figure's runtime."""
    k = (mouse, day)
    if k not in _OF_CACHE:
        stem = f'{SRC}M{mouse}/D{day}/OF1/sub-M{mouse}_ses-D{day}_typ-OF1'
        _OF_CACHE[k] = (nap.load_file(f'{stem}_beh.nwb'),
                        nap.load_file(f'{stem}_srt-kilosort4_clusters.npz'))
    return _OF_CACHE[k]


def of_rate_map(cid, mouse, day, df):
    beh, clusters = _of_session(mouse, day)
    lag = df[df.cluster_id == cid].travel.values[0]
    pos = np.stack([beh['P_x'], beh['P_y']], axis=1)
    bl = compute_travel_projected(['P_x', 'P_y'], pos, pos, lag)
    pl = np.stack([bl['P_x'], bl['P_y']], axis=1)
    tc = nap.compute_2d_tuning_curves(nap.TsGroup([clusters[cid]]), pl,
                                      nb_bins=(40, 40))[0][0]
    tc = gaussian_filter_nan(_fill_nans_from_neighbors(tc), sigma=(2.5, 2.5))
    return np.clip(tc, 0, np.nanpercentile(tc, 99))


def pick_cells(day):
    sub = _cls[(_cls.mouse == 25) & (_cls.day == day)]
    gc = sub[sub.cell_type == 'GC'].copy()
    ngs = sub[sub.cell_type == 'NG'].copy()
    tot = len(gc) + len(ngs)
    n_gc = max(round(len(gc) / tot * RM_N), min(1, len(gc))) if tot else 0
    gc = gc.sort_values('grid_score_best', ascending=False).head(n_gc)
    ngs = ngs.sort_values('SC_y')
    if len(ngs) > RM_N - n_gc:
        ngs = ngs.iloc[np.linspace(0, len(ngs) - 1, RM_N - n_gc).astype(int)]
    ngs = ngs.sort_values('spatial_information_score_best', ascending=False)
    gc = gc.assign(_t='GC'); ngs = ngs.assign(_t='NGS')
    return pd.concat([gc, ngs]).reset_index(drop=True), len(gc), len(ngs)


# ── H, I: two example sessions, each as slice + raster + PC1 + agreement ─────
# J and L report the mediolateral gradient in following as a correlation. These
# rows show the thing the correlation is computed on, in two sessions. Cells are
# ordered by mediolateral position rather than by PC1 loading, so if position
# organises following at all, the medial (left) end of each raster should track
# the population state more tightly than the lateral end.
#
# M25 D25 is kept for continuity with the rate maps in panel A, which are all
# M25; it is anchored on most trials, so its raster is mostly one colour. M26 D19
# was added because it has what M25 D25 lacks: balanced states (PC1 positive on
# half the trials), 13 sustained transitions, and four times the cells -- with
# the following concentrated on the medial shank (mean agreement 0.44 medial
# against 0.08 lateral, rho = -0.55). It was chosen by ranking every clean
# per-shank session (>= 3 shanks; M20/M21 positions carry per-cell jitter and
# were excluded) on sustained transitions, state balance and the within-session
# mediolateral rho; it is the second session of M26, the day after the Figure 1
# example. M28 D25 was added as a third because, with every region counted, its
# most medial shank looks unresponsive -- the region strip shows why.
#
# TRANSITIONS are where z-scored PC1 changes sign and holds the new sign for at
# least three trials, as in Figure 8A-D. They used to come from a 9-trial median
# filter on the fraction of cells called anchored, which is a different quantity
# from the PC1 strip the dashed lines are drawn across.
from scipy.ndimage import median_filter  # noqa: F401  (kept for other callers)
from spatial_manifolds.anchoring import (ANCH_COLOR as ANCH_C,
                                         NONANCH_COLOR as NONANCH_C,
                                         load_session_labels)

# Two per row. D is M29 D23, chosen to add a fourth animal (rank 8 of 27 by
# MEC-only rho, -0.19). The stronger within-MEC alternative was M25 D18 (all four
# shanks in MEC, rank 5). M29's other sessions are flat or reversed, which the
# caption says.
EXAMPLES = [('A', 25, 25), ('B', 26, 19), ('C', 28, 25), ('D', 29, 25)]
TA_ = ListedColormap([NONANCH_C, ANCH_C])
NORM_ = BoundaryNorm([-.5, .5, 1.5], TA_.N)
_rr_all = pd.read_csv(f'{PS}/pc1_by_region.csv')


def pc1_transitions(pc1, min_run=3):
    """z-scored PC1 and the trials (1-based) where it changes sign for >= min_run."""
    z = (pc1 - np.nanmean(pc1)) / np.nanstd(pc1)
    s = np.where(np.nan_to_num(z) >= 0, 1, -1)
    runs = []
    for k, v in enumerate(s):
        if runs and runs[-1][0] == v:
            runs[-1][2] = k
        else:
            runs.append([v, k, k])
    merged = []
    for v, a_, b_ in runs:
        if merged and (b_ - a_ + 1 < min_run or merged[-1][0] == v):
            merged[-1][2] = b_
        else:
            merged.append([v, a_, b_])
    return z, [m[1] + 1 for m in merged[1:]]


MIN_SHANK_SP = 10          # classified cells for a shank's point in Figure 4D
PROBE_JITTER = 55.0        # um, display spread of cells across their shank
SHANK_X = [16 + 250 * k for k in range(4)]


def probe_axes(ax, mo, dy, cells):
    """Draw the NP2014 probe on ax, every contact light and the session's recorded
    contacts darker, shanks re-ordered medial -> lateral. Returns each cell's
    display x and y (um from the tip), its device shank and its medial->lateral
    rank."""
    import probeinterface as pi
    from matplotlib.collections import PatchCollection
    A_ = pd.read_csv(f'{SRC}M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv')
    A_['s'] = A_.device_contact_id.str.extract(r's(\d)e')[0].astype(int)
    order = A_.groupby('s').coord_SCs_x.apply(lambda v: v.abs().mean()).sort_values().index
    rank_of = {int(s_): r_ for r_, s_ in enumerate(order)}
    probe = pi.get_probe('imec', 'NP2014')
    xy = np.round(probe.contact_positions).astype(int)
    on = set(zip(A_.coord_probe_x.round().astype(int), A_.coord_probe_y.round().astype(int)))
    act = np.array([(x_, y_) in on for x_, y_ in xy])
    dev = np.clip(np.round((xy[:, 0] - 16) / 250).astype(int), 0, 3)
    X = np.array([rank_of[d_] for d_ in dev]) * 250 + (xy[:, 0] - 250 * dev)
    ya = xy[act, 1]
    y0, y1 = ya.min() - 260, ya.max() + 260
    for r_ in range(4):                                 # the shank outlines
        ax.add_patch(Rectangle((250 * r_ - 20, y0 - 50), 72, y1 - y0 + 100, facecolor='0.97',
                               edgecolor='0.6', lw=.4, zorder=1))
    vis = (xy[:, 1] > y0) & (xy[:, 1] < y1)
    for sel, col in ((~act & vis, '0.86'), (act, '0.5')):
        ax.add_collection(PatchCollection([Rectangle((x_ - 6, y_ - 6), 12, 12)
                                           for x_, y_ in zip(X[sel], xy[sel, 1])],
                                          facecolor=col, lw=0, zorder=1.2))
    cdev = np.clip(np.round((cells.coord_probe_x.values - 16) / 250).astype(int), 0, 3)
    crank = np.array([rank_of[d_] for d_ in cdev])
    cx = crank * 250 + (cells.coord_probe_x.values - 250 * cdev)
    ax.set_xlim(-90, 3 * 250 + 32 + 90); ax.set_ylim(y0, y1)
    ax.set_xticks(SHANK_X); ax.set_xticklabels([f'S{k + 1}' for k in range(4)], fontsize=6.5)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_xlabel('shank, medial → lateral', fontsize=7)
    ax.set_ylabel('µm from probe tip', fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    return cx, cells.coord_probe_y.values, cdev, crank


def example_row(cell, letter, mo, dy, merge_mec=False):
    # wspace is wider than the rows above because this one carries a colorbar
    # on the slice panel: the bar's tick labels and the raster's rotated y-label
    # both float outward by a fixed point padding that no gridspec setting
    # accounts for, so a gap that looks ample between the boxes collides in ink.
    # [raster | PC1] tight together on the left, the slice (+ boxplot strip) on
    # the right. The slice is drawn at equal aspect, so it is height-limited
    # and would float in the middle of its slot; it is anchored to the right
    # edge instead, which closes the gap that otherwise opens after PC1.
    gR = cell.subgridspec(1, 2, width_ratios=[2.55, 1.15], wspace=.30)
    gLft = gR[0].subgridspec(1, 2, width_ratios=[2.25, .30], wspace=.05)
    z = load_session_labels(mo, dy)
    ids = [int(c) for c in z['cluster_id']]
    cs = (C[(C.mouse == mo) & (C.day == dy)].dropna(subset=['coord_SCs_x'])
          .assign(ml=lambda d: d.coord_SCs_x.abs()).set_index('cluster_id'))
    # medial -> lateral by shank, then DORSAL -> VENTRAL within the shank
    # (coord_SCs_y grows ventrally: visual cortex sits near 2240, MEC 2550-2815),
    # so each shank's region strip reads like the shank itself
    # cells whose labels are NaN on every trial carry no anchoring call at all;
    # they are dropped from every panel of the row (raster, strip, slice, boxes)
    _all_nan = np.isnan(z['labels']).all(axis=1)
    nolab = {c for i, c in enumerate(ids) if _all_nan[i]}
    keep = sorted([(i, c) for i, c in enumerate(ids)
                   if c in cs.index and not _all_nan[i]],
                  key=lambda t: (cs.ml[t[1]], cs.coord_SCs_y[t[1]]))
    rows = [i for i, _ in keep]
    cid = [c for _, c in keep]
    mls = np.array([cs.ml[c] for c in cid])
    rcls = np.array([region_class(cs.brain_region[c]) for c in cid])
    L = z['labels'][rows]
    frac = np.asarray(z['frac_anch'], float)
    n_tr = L.shape[1]
    pcz, tr = pc1_transitions(np.asarray(z['pc1'], float))
    rr = _rr_all.query('mouse == @mo and day == @dy').set_index('cluster_id')

    # the probe, not a best-fit slice: the 4-shank Neuropixels 2.0 (NP2014) with
    # the session's recorded contacts darker than the inactive ones, as Figure
    # S22A draws it, and each cell at its localised position coloured by agreement
    # ABOVE CHANCE -- |r| minus the mean |r| of the cell's own circular-shift
    # null, so absolute is compared with absolute (subtracting an absolute null
    # from a signed r is the error that retired two earlier claims). Centred on
    # chance, so cells below their own null show too. Grid cells are triangles
    # with a thin dark edge (the user did not want them outlined in colour). Shanks are drawn medial -> lateral (S1 medial), the
    # order of the raster beside it, whatever the device's own numbering.
    gSl = gR[1].subgridspec(2, 1, height_ratios=[.42, 1], hspace=.08)
    axbx = fig.add_subplot(gSl[0])
    axsl = fig.add_subplot(gSl[1])
    cells = C[(C.mouse == mo) & (C.day == dy)].dropna(subset=['coord_probe_x', 'coord_probe_y'])
    cells = cells[~cells.cluster_id.isin(nolab)]
    ex = np.array([rr.excess.get(c, np.nan) for c in cells.cluster_id])
    sx, sy, dsh, rank = probe_axes(axsl, mo, dy, cells)
    rng_ = np.random.default_rng(mo * 100 + dy)
    px = sx + rng_.uniform(-PROBE_JITTER, PROBE_JITTER, len(sx))
    okc = np.isfinite(ex)
    hi = max(abs(np.nanpercentile(ex, 2)), abs(np.nanpercentile(ex, 98)))
    norm_ = TwoSlopeNorm(vcenter=0.0, vmin=-hi, vmax=hi)
    # DIVERGING, white at chance. PRGn because the obvious alternatives each
    # collide with a colour this figure already uses: BrBG ends in the
    # non-anchored teal, PiYG in near-magenta, RdBu in the grid/non-grid red
    # and blue, and PuOr's orange in the interneuron orange.
    is_gc = (cells.cell_class_of1 == 'GC').values
    axsl.scatter(px[~okc & ~is_gc], sy[~okc & ~is_gc], s=5, color='0.72', lw=0, zorder=2)
    axsl.scatter(px[okc & ~is_gc], sy[okc & ~is_gc], c=ex[okc & ~is_gc], s=9, cmap='PRGn',
                 norm=norm_, lw=.25, edgecolors='0.5', zorder=3)
    axsl.scatter(px[~okc & is_gc], sy[~okc & is_gc], s=26, marker='^', color='0.72',
                 edgecolors='0.15', lw=.5, zorder=4)
    axsl.scatter(px[okc & is_gc], sy[okc & is_gc], c=ex[okc & is_gc], s=26, marker='^',
                 cmap='PRGn', norm=norm_, edgecolors='0.15', lw=.5, zorder=4)
    cax = axsl.inset_axes([1.04, .0, .045, .42])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm_, 'PRGn'), cax=cax)
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1); cb.outline.set_visible(False)
    cax.set_title('above\nchance', fontsize=5.5, pad=2, loc='left')
    n_gc = int(is_gc.sum())
    axsl.text(1.03, 1.0, f'▲ grid cell ({n_gc})\n● other cell\ngrey: locked\n■ recorded contact\n□ inactive',
              transform=axsl.transAxes, fontsize=5.5, va='top', ha='left', color='0.25',
              linespacing=1.35)

    # agreement above chance, one box per shank directly over the shank
    bx_vals = [ex[(rank == j) & okc] for j in range(4)]
    axbx.boxplot(bx_vals, positions=SHANK_X, widths=90, showfliers=False,
                 patch_artist=True, manage_ticks=False,
                 medianprops=dict(color='k', lw=1.3),
                 boxprops=dict(facecolor='0.85', edgecolor='0.35', lw=.7),
                 whiskerprops=dict(color='0.35', lw=.7),
                 capprops=dict(color='0.35', lw=.7))
    axbx.axhline(0, color='0.5', lw=.7, ls=':', zorder=1)
    axbx.set_xlim(axsl.get_xlim())
    axbx.set_xticks([])
    axbx.set_ylabel('above\nchance', fontsize=6, labelpad=1)
    axbx.tick_params(labelsize=5.5)
    axbx.spines[['top', 'right', 'bottom']].set_visible(False)

    # raster, one block per shank, each under a strip giving the cell's region
    shanks = np.unique(mls)
    counts = [int((mls == u).sum()) for u in shanks]
    HR = [.055, 1]
    gS = gLft[0].subgridspec(2, len(shanks), height_ratios=HR,
                           width_ratios=counts, wspace=.035, hspace=.025)
    strip_rgb = np.array([(GREY_OTHER,) * 3] +
                         [(REGION_GREY[k],) * 3 for k in (1, 2, 3, 4, 5)])
    for j, u in enumerate(shanks):
        m = mls == u
        axs = fig.add_subplot(gS[0, j])
        # merge_mec: one grey for all of MEC, as Figure 4's key has no layer split
        _rc = np.where(rcls[m] == 4, 3, rcls[m]) if merge_mec else rcls[m]
        axs.imshow(strip_rgb[_rc][None, :, :], aspect='auto',
                   interpolation='nearest')
        axs.set_xticks([]); axs.set_yticks([])
        for sp_ in axs.spines.values():
            sp_.set_visible(False)
        # a shank with very few cells is too narrow for its label, which then
        # runs into the next one; the S label below still identifies it
        if m.sum() >= .06 * len(mls):
            axs.set_title(f'{u:.0f} µm', fontsize=6, color='0.25', pad=2)
        if j == 0:
            axs.set_ylabel('region', fontsize=6, rotation=0, ha='right',
                           va='center', labelpad=2)
            axs.annotate(f'M{mo} D{dy}, {len(rows)} cells',
                         (0, 1), xycoords='axes fraction', xytext=(0, 13),
                         textcoords='offset points', fontsize=7.4, ha='left',
                         va='bottom')
        ax = fig.add_subplot(gS[1, j])
        ax.imshow(L[m].T, aspect='auto', cmap=TA_, norm=NORM_,
                  interpolation='nearest', extent=[0, m.sum(), n_tr, 1])
        for tb in tr:
            ax.axhline(tb, color='0.1', lw=.7, ls='--', zorder=4)
        ax.set_xticks([])
        ax.set_xlabel(f'S{j + 1}', fontsize=7, weight='bold', labelpad=2)
        if j == 0:
            ax.set_ylabel('Trial', fontsize=8)
            ax.tick_params(labelsize=7)
        else:
            ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)

    # z-scored PC1, on the same rows as the raster (same height split)
    # the row letter, at the raster's top-left now that the raster leads
    _pos = gLft[0].get_position(fig)
    fig.text(_pos.x0 - .022, _pos.y1 + .012, letter, fontsize=10, weight='bold',
             va='bottom', ha='right')
    gP = gLft[1].subgridspec(2, 1, height_ratios=HR, hspace=.025)
    axp = fig.add_subplot(gP[1])
    pc = np.nan_to_num(pcz)
    y = np.arange(1, len(pc) + 1)
    axp.fill_betweenx(y, 0, pc, where=pc >= 0, color=ANCH_C, lw=0,
                      edgecolor='none', interpolate=True)
    axp.fill_betweenx(y, 0, pc, where=pc < 0, color=NONANCH_C, lw=0,
                      edgecolor='none', interpolate=True)
    axp.axvline(0, color='0.4', lw=.7)
    for tb in tr:
        axp.axhline(tb, color='0.1', lw=.8, ls='--', zorder=4)
    axp.set_ylim(n_tr, 1); axp.set_xticks([])
    axp.tick_params(labelleft=False, left=False)
    # labelled underneath: above, it collides with the last shank's µm label
    # now that PC1 sits tight against the raster
    axp.set_xlabel('PC1\n(z)', fontsize=7, labelpad=2)
    for sp_ in axp.spines.values():
        sp_.set_visible(False)

    # The per-session statistics the caption quotes. The panel that drew them
    # (agreement against position) was removed once the boxplot strip above
    # the slice showed the same per-shank distributions on the slice's own axis.
    rv = np.array([rr.r.get(c, np.nan) for c in cid])
    ok = np.isfinite(rv)
    per_shank = []
    for u in shanks:
        m = (mls == u) & ok
        if m.sum() >= 3:
            per_shank.append((u, rv[m].mean(), int(m.sum())))
    rho = spearmanr(mls[ok], rv[ok])
    mec_ = ok & np.isin(rcls, (3, 4))
    rho_m = spearmanr(mls[mec_], rv[mec_])
    print(f'  example {letter} M{mo}D{dy}: {len(rows)} cells in raster, '
          f'{int(ok.sum())} with agreement, {len(shanks)} shanks, '
          f'span {mls.max() - mls.min():.0f} um, {(frac > .5).mean() * 100:.0f}% '
          f'anchored, {len(tr)} PC1 transitions, rho={rho[0]:+.3f} p={rho[1]:.3g}; '
          'per-shank agreement medial->lateral: '
          + ', '.join(f'{v:.2f} (n={n})' for _, v, n in per_shank)
          + f' | MEC only: rho={rho_m[0]:+.3f} p={rho_m[1]:.3g}; per shank '
          + ', '.join(f'{rv[(mls == u) & mec_].mean():.2f}' for u in shanks
                      if ((mls == u) & mec_).sum() >= 3))
    print('     region classes by shank: ' + '; '.join(
        f'{u:.0f}:' + ','.join(f'{k}={int(((mls == u) & (rcls == k)).sum())}'
                               for k in range(6) if ((mls == u) & (rcls == k)).any())
        for u in shanks))
    return axbx


# ── panels ──────────────────────────────────────────────────────────────────
# Each panel is a function of the axes (or gridspec cell) it draws into, so the
# main figure and its supplement (Figure S9, the panels the merge left out) can
# each take what they need.

def session_schematic(ax, letter):
    """The recording day, top to bottom: the 1 m x 1 m open-field arena (OF1),
    the virtual-reality task (a head-fixed mouse on a cylinder before the
    screens), the arena again (OF2). OF1 is marked: every map in the rate-map
    panel comes from it, the session that PRECEDES the task, which is what
    makes cell identity independent of the behaviour it is later used to
    explain."""
    from matplotlib.patches import Circle, Ellipse, Polygon, Arc
    ax.set_xlim(0, 1); ax.set_ylim(-.55, 5.2); ax.set_aspect('equal'); ax.axis('off')
    ax.set_anchor('W')
    rng = np.random.default_rng(5)

    def arena(yc, hot, name):
        x0, y0, w = .2, yc - .3, .6
        ax.add_patch(Rectangle((x0, y0), w, w, facecolor='white',
                               edgecolor=COL_GC if hot else '0.35', lw=1.4 if hot else .9, zorder=2))
        st = np.cumsum(rng.normal(0, .035, (160, 2)), axis=0)
        st = (st - st.min(0)) / np.ptp(st, axis=0) * (w - .1) + [x0 + .05, y0 + .05]
        ax.plot(st[:, 0], st[:, 1], color='0.6', lw=.45, zorder=3)
        ax.text(.5, y0 + w + .05, name, ha='center', va='bottom', fontsize=6.6,
                weight='bold', color=COL_GC if hot else '0.25')
        ax.annotate('', (x0, y0 - .06), (x0 + w, y0 - .06),
                    arrowprops=dict(arrowstyle='<->', color='0.4', lw=.6, shrinkA=0, shrinkB=0))
        ax.text(.5, y0 - .09, '1 m', ha='center', va='top', fontsize=5.6, color='0.35')

    arena(4.35, True, 'OF 1')
    # VR: screens wrapped round a head-fixed mouse on a cylinder, side view
    yc = 2.5
    ax.add_patch(Arc((.62, yc + .02), .7, .66, theta1=-70, theta2=70, color='0.25', lw=2.2, zorder=2))
    for k_ in range(5):                                       # gratings on the screen
        th = np.deg2rad(-50 + 25 * k_)
        ax.plot([.62 + .35 * np.cos(th)] * 2, [yc + .02 + .33 * np.sin(th) - .03,
                                               yc + .02 + .33 * np.sin(th) + .03],
                color='0.55', lw=1.2, zorder=3)
    ax.add_patch(Circle((.45, yc - .2), .13, facecolor='0.9', edgecolor='0.35', lw=.8, zorder=2))
    ax.add_patch(Ellipse((.43, yc + .0), .3, .14, facecolor='0.55', edgecolor='none', zorder=3))
    ax.add_patch(Ellipse((.6, yc + .03), .12, .09, facecolor='0.55', edgecolor='none', zorder=3))
    ax.add_patch(Polygon([(.6, yc + .06), (.56, yc + .13), (.63, yc + .09)], color='0.45', zorder=4))
    ax.plot([.28, .2, .16], [yc - .01, yc - .06, yc - .02], color='0.55', lw=.8, zorder=3)
    ax.plot([.58, .58], [yc + .07, yc + .2], color='0.3', lw=1.4, zorder=4)   # head-post
    ax.text(.5, yc + .42, 'VR task', ha='center', va='bottom', fontsize=6.6, color='0.25',
            weight='bold')
    arena(.7, False, 'OF 2')
    for y_a, y_b in ((3.78, 3.25), (1.82, 1.4)):
        ax.annotate('', (.5, y_b), (.5, y_a),
                    arrowprops=dict(arrowstyle='-|>', color='0.45', lw=.8))
    ax.text(.5, 5.2, 'one recording day', ha='center', va='top', fontsize=6.0,
            style='italic', color='0.35')
    ax.text(.5, -.1, f'maps in {letter}\nare from OF 1', ha='center', va='top',
            fontsize=5.8, color=COL_GC)


def panel_ratemaps(spec_seq, spec_maps, letter):
    """The gradient shown as OF1 rate maps, four M25 sessions medial -> lateral."""
    g0 = spec_maps.subgridspec(1, len(RM_SESSIONS), wspace=.07)
    for bi, (day, ml) in enumerate(RM_SESSIONS):
        cells, n_gc, n_ngs = pick_cells(day)
        gb = g0[bi].subgridspec(RM_N // RM_COLS, RM_COLS, wspace=.05, hspace=.05)
        for ci, (_, row) in enumerate(cells.iterrows()):
            axm = fig.add_subplot(gb[ci])
            try:
                axm.imshow(of_rate_map(int(row.cluster_id), 25, day, cells), cmap='viridis', aspect='equal')
            except Exception:
                axm.set_facecolor('0.92')
            axm.set_xticks([]); axm.set_yticks([])
            for _spine in axm.spines.values():
                _spine.set_color(COL_GC if row._t == 'GC' else COL_NGS)
                _spine.set_linewidth(1.1)
            if ci == 1:
                axm.set_title(f'D{day}  ML≈{ml} µm\n{n_gc} grid + {n_ngs} non-grid',
                              fontsize=6.6, pad=4)
        print(f'  M25 D{day} (ML≈{ml}): {n_gc} grid + {n_ngs} non-grid drawn')
    axs_ = fig.add_subplot(spec_seq)
    session_schematic(axs_, letter)
    _p0, _p1 = g0[0].get_position(fig), g0[len(RM_SESSIONS) - 1].get_position(fig)
    fig.text(_p0.x0 - .028, _p0.y1 + .02, letter, fontsize=10, weight='bold', va='bottom')
    fig.text((_p0.x0 + _p1.x1) / 2, _p0.y0 - .008,
             'medial  ←' + '—' * 46 + '→  lateral', fontsize=6.4, ha='center',
             va='top', style='italic', color='0.3')
    return axs_


def panel_slice(spec, letter, name, mouse, first):
    """A small 3D view over the mouse's best-fit slice, cells by identity.
    Three individual mice rather than a pooled fit: the pooled slice averaged
    over animals whose probes sat at different mediolateral positions, which is
    the variable the panel is about."""
    gg = spec.subgridspec(2, 1, height_ratios=[.34, 1], hspace=.04)
    axb = fig.add_subplot(gg[0])
    try:
        png = render_brain(mouse, f'{RENDERS}/brain_m{mouse}.png')
        axb.imshow(trim_white(plt.imread(png)))
    except Exception as e:
        axb.text(.5, .5, f'render unavailable\n{type(e).__name__}', ha='center',
                 va='center', transform=axb.transAxes, fontsize=6.5, color='0.5')
    axb.axis('off')
    _lp(axb, letter)
    axb.set_title(name, fontsize=8.5, loc='left')
    ax = fig.add_subplot(gg[1])
    gc_p, ngs_p, oth_p = get_mouse_pts(mouse)
    contacts = get_mouse_contacts(mouse)
    allp = np.concatenate([p for p in (gc_p, ngs_p, oth_p) if len(p)])
    fit = fit_best_slice(contacts, extent_pts=allp)
    u_vals, v_vals, grid = sample_region_grid(fit)
    # both axes are micrometres, so the slice is drawn to scale
    ax.imshow(grid, origin='lower', aspect='equal', cmap=region_cmap, norm=region_norm,
              extent=[u_vals[0], u_vals[-1], v_vals[0], v_vals[-1]], zorder=1)
    ax.set_aspect('equal', adjustable='box')
    rng = np.random.default_rng(0)
    xs, ys, cs = [], [], []
    for p, col in ((oth_p, COL_OTHER), (ngs_p, COL_NGS), (gc_p, COL_GC)):
        if not len(p):
            continue
        u, v = fit['project'](p)
        xs.append(u); ys.append(v); cs.append(np.full(len(u), col))
    xs, ys, cs = (np.concatenate(a) for a in (xs, ys, cs))
    o = rng.permutation(len(xs))
    ax.scatter(xs[o], ys[o], color=cs[o], s=3.0, alpha=CELL_ALPHA, lw=0, zorder=3)
    ax.set_xlabel('medial–lateral (µm)', fontsize=8)
    if first:
        ax.set_ylabel('dorsal–ventral (µm)', fontsize=8)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.tick_params(labelsize=6.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(f'{len(gc_p)} grid, {len(ngs_p)} non-grid\n{fit["tilt_deg"]:+.1f}° from coronal',
                 fontsize=6.4, loc='left')


def panel_grid_axis(ax, axis, lab, letter, ylabel=True, shank_points=False):
    """Fraction of spatial cells that are grid cells along one axis.
    shank_points: one point per shank of every session (each shank of a
    multi-shank session separately) with at least MIN_SHANK_SP spatial cells,
    at the shank's position. Shanks are grouped on mediolateral position to
    50 um, which absorbs the per-cell jitter of the M20/M21 positions."""
    if shank_points:
        q = sp.assign(_sh=(sp[axis] / 50).round() * 50)
        q = (q.groupby(['mouse', 'day', '_sh'])
             .agg(n=('gc', 'size'), f=('gc', 'mean'), x=(axis, 'mean')).reset_index())
        q = q[q.n >= MIN_SHANK_SP]
        ax.scatter(q.x + np.random.default_rng(3).uniform(-12, 12, len(q)), q.f, s=9,
                   color='0.6', alpha=.6, lw=0, zorder=1)
        print(f'  {letter}: {len(q)} shank points (>= {MIN_SHANK_SP} spatial cells), '
              f'{q.groupby(["mouse", "day"]).ngroups} sessions')
    x, y, e = binned(sp, axis, 'gc')
    ax.fill_between(x, y - e, y + e, color=COL_GC, alpha=.25, lw=0, edgecolor='none')
    ax.plot(x, y, color=COL_GC, lw=1.8)
    ws, p = within_session(sp, axis, 'gc')
    ax.set_xlabel(lab, fontsize=8)
    if ylabel:
        ax.set_ylabel('fraction grid\n(of spatial cells)', fontsize=8)
    ax.set_ylim(0, max(ax.get_ylim()[1], .02) * (1.05 if shank_points else 1.22))
    _lp(ax, letter)
    ax.set_title(f'within session rho = {np.median(ws):+.3f}\n'
                 f'p = {p:.3g}  ({len(ws)} sessions)', fontsize=7, loc='left')
    print(f'  {letter} grid vs {axis}: within-session median rho {np.median(ws):+.3f}, '
          f'p = {p:.3g}, {(np.array(ws) < 0).sum()} of {len(ws)} negative')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)


IDENT = [('grid', 'grid', COL_GC), ('non-grid spatial', 'non-grid spatial', COL_NGS),
         ('non-spatial', 'non-spatial', '#888888'), ('putative interneuron', 'interneuron', '#d95f02')]


def mixed_ml(df, y):
    """Within-session mediolateral slope of y (per 100 um) from a linear mixed
    model with MOUSE as the top-level random effect: random intercept and
    random within-session slope per mouse, session intercepts nested within
    mouse. Position is split into its within-session part (the cell's offset
    from its session's mean position, the one tested) and the session mean
    (between-session placement, a nuisance term), so probe placement differing
    between animals cannot produce the slope, and sessions from one mouse are
    not counted as independent replicates. A 0/1 identity is fitted as a linear
    probability model. Returns slope, p, n mice, n sessions."""
    import statsmodels.formula.api as smf
    d = df.dropna(subset=['ml', y]).copy()
    d['sess'] = d.mouse.astype(str) + '_' + d.day.astype(str)
    sm = d.groupby('sess').ml.transform('mean')
    d['ml_w'], d['ml_b'] = (d.ml - sm) / 100, (sm - sm.mean()) / 100
    # 'sess' is a string column, so patsy treats it as categorical; C(sess)
    # would collide with this module's cell table, also called C
    m = smf.mixedlm(f'{y} ~ ml_w + ml_b', d, groups=d['mouse'], re_formula='1 + ml_w',
                    vc_formula={'sess': '0 + sess'}).fit(reml=True, method='lbfgs')
    assert m.converged, f'mixed_ml({y}) did not converge'
    return m.params['ml_w'], m.pvalues['ml_w'], d.mouse.nunique(), d.sess.nunique(), \
        (m, sm.mean())


def draw_mixed(ax, fit, lo, hi, col, band_alpha=.3):
    """The mixed model's within-session slope as a line, drawn through the
    fitted value of a session at the mean session position (ml_b = 0) and
    spanning lo..hi; band, 95% CI of that line from the fixed effects."""
    m, x0 = fit
    x = np.linspace(lo, hi, 50); dx = (x - x0) / 100
    b0, b1 = m.fe_params['Intercept'], m.fe_params['ml_w']
    V = m.cov_params().loc[['Intercept', 'ml_w'], ['Intercept', 'ml_w']].values
    y = b0 + b1 * dx
    se = np.sqrt(V[0, 0] + 2 * dx * V[0, 1] + dx ** 2 * V[1, 1])
    ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=col, alpha=band_alpha, lw=0,
                    edgecolor='none', zorder=3)
    ax.plot(x, y, color=col, lw=1.6, zorder=4)


def running_fraction(x, y, half=100., n=40, lo=None, hi=None):
    g = np.linspace(x.min() if lo is None else lo, x.max() if hi is None else hi, n)
    m = np.full(n, np.nan); e = np.full(n, np.nan)
    for i, g_ in enumerate(g):
        v = y[np.abs(x - g_) <= half]
        if len(v) >= 10:
            m[i], e[i] = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    return g, m, e


def panel_composition(spec, letter):
    """The four identities of Figure 3 as fractions of all classified MEC cells
    (the cells of the unit table, those analysed in the task) along the
    mediolateral axis: line, cells pooled in quantile bins with SEM; points,
    one per shank (each shank of a multi-shank session separately, >=
    MIN_SHANK_SP classified cells). Within-session rho on the sessions spanning
    two shank pitches, as everywhere in this figure."""
    q0 = mec.dropna(subset=['identity']).copy()
    q0['_sh'] = (q0.ml / 50).round() * 50
    gq = spec.subgridspec(1, 4, wspace=.38)
    sel = multi_shank(q0, 'ml')
    for k, (idn, nm, col) in enumerate(IDENT):
        ax = fig.add_subplot(gq[0, k])
        q0['y'] = (q0.identity == idn).astype(float)
        sh = (q0.groupby(['mouse', 'day', '_sh'])
              .agg(n=('y', 'size'), f=('y', 'mean'), x=('ml', 'mean')).reset_index())
        sh = sh[sh.n >= MIN_SHANK_SP]
        ax.scatter(sh.x + np.random.default_rng(3).uniform(-12, 12, len(sh)), sh.f, s=5,
                   color=col, alpha=.3, lw=0, zorder=1)
        # the fitted line spans the most medial to the most lateral shank point
        # (>= MIN_SHANK_SP cells; the 5 cells of M28 D18 at 2910 um are too few
        # to carry a point)
        ws = []
        for kk, g in q0.groupby(['mouse', 'day']):
            if kk in sel and len(g) >= 20 and g.y.nunique() > 1:
                ws.append(spearmanr(g.ml, g.y)[0])
        ws = np.array(ws); p = wilcoxon(ws).pvalue
        b_, pm_, nmo, nse, fit_ = mixed_ml(q0, 'y')
        # line, the mixed model's within-session slope (mouse random effect)
        draw_mixed(ax, fit_, sh.x.min(), sh.x.max(), col)
        ax.set_title(f'{nm}\n{100 * b_:+.1f}% per 100 µm, p = {pm_:.2g}', fontsize=6.3,
                     loc='left', color=col, pad=2)
        print(f'  {letter} {idn}: fraction {q0.y.mean():.3f}; within-session median rho '
              f'{np.median(ws):+.3f}, {(ws < 0).sum()} of {len(ws)} negative, p = {p:.3g}; '
              f'{len(sh)} shank points; MIXED (mouse) within-session slope {100 * b_:+.2f}% '
              f'per 100 um, p = {pm_:.3g}, {nmo} mice, {nse} sessions')
        ax.set_ylim(0, .5 if idn == 'grid' else None)
        ax.set_xlim(2980, 3820)
        ax.set_xticks([3000, 3400, 3800])
        ax.tick_params(labelsize=5.8, length=2, pad=1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('medial–lateral (µm)', fontsize=6.5, labelpad=1)
        if k == 0:
            ax.set_ylabel('fraction of\nMEC cells', fontsize=6.5)
            _lp(ax, letter, dx=-.5, dy=1.25)
            ax0 = ax
    print(f'  {letter}: {len(q0)} classified MEC cells')
    return ax0


def panel_layer(ax, letter):
    """Grid fraction superficial against deep, per session."""
    rows = []
    for (mo, dy), g in sp.groupby(['mouse', 'day']):
        a, b = g[g.sup], g[~g.sup]
        if len(a) >= 10 and len(b) >= 10:
            rows.append((a.gc.mean(), b.gc.mean()))
    R = np.array(rows)
    pl = wilcoxon(R[:, 0] - R[:, 1]).pvalue
    print(f'  layer: grid fraction superficial {R[:, 0].mean():.1%} vs deep {R[:, 1].mean():.1%}, '
          f'p = {pl:.3g}, higher superficially in {(R[:, 0] > R[:, 1]).sum()} of {len(R)}')
    for i in range(len(R)):
        ax.plot([0, 1], R[i], color='0.75', lw=.7, zorder=2)
    for j, col in enumerate((COL_GC, '0.45')):
        ax.errorbar(j, R[:, j].mean(), yerr=R[:, j].std(ddof=1) / np.sqrt(len(R)),
                    color=col, marker='o', ms=6, lw=1.6, capsize=4, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['superficial\n(L1–3)', 'deep\n(L5–6)'], fontsize=7)
    ax.set_xlim(-.4, 1.4)
    ax.set_ylabel('fraction grid', fontsize=8)
    _lp(ax, letter)
    ax.set_title(f'higher in {int((R[:, 0] > R[:, 1]).sum())}/{len(R)} sessions\n'
                 f'p = {pl:.3g}', fontsize=7, loc='left')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)


def region_legend(y, cells=False, simple=False, anchor=None):
    """simple: Figure 4's key, MEC as one region and no hippocampus (none of its
    example sessions has hippocampal cells)."""
    keys = ((2, 'Pre/parasubiculum'), (3, 'MEC'), (5, 'Visual cortex')) if simple else \
        tuple(zip((1, 2, 3, 4, 5), REGION_NAMES))
    h = ([Patch(facecolor=str(REGION_GREY[k]), edgecolor='0.6', lw=.4, label=l) for k, l in keys]
         + [Patch(facecolor=str(GREY_OTHER), edgecolor='0.6', lw=.4, label='other / unassigned')])
    if cells:
        h += [Patch(facecolor=c, label=l) for l, c in
              (('grid cell', COL_GC), ('non-grid spatial', COL_NGS), ('other', COL_OTHER))]
    if anchor is not None:                  # a compact key at a given figure point
        fig.legend(handles=h, loc='upper left', bbox_to_anchor=anchor, ncol=1, fontsize=6.0,
                   frameon=False, handlelength=1.0, handleheight=.9, labelspacing=.3,
                   borderpad=0, title='region (raster strips)', title_fontsize=6.0,
                   alignment='left')
        return
    fig.legend(handles=h, loc='center', bbox_to_anchor=(.5, y), ncol=len(h), fontsize=6.6,
               frameon=False, handlelength=1.0, handleheight=1.1, columnspacing=1.3,
               borderpad=0, title='region (raster strips' + (', slices' if cells else '') + ')',
               title_fontsize=6.6)


foll = mec.dropna(subset=['r'])
N_ML_SESS = len(within_session(foll, 'ml', 'r')[0])
_EX = pd.read_csv(f'{PS}/pc1_by_region.csv')[['mouse', 'day', 'cluster_id', 'excess']]
follx = foll.merge(_EX, on=['mouse', 'day', 'cluster_id'], how='left')


def panel_ml(ax, letter):
    """The pooled mediolateral gradient in agreement ABOVE CHANCE (the quantity
    the example probes and boxplots show), not raw agreement with the axis."""
    # one point per shank (session x position to 50 um, >= MIN_SHANK_SP MEC cells
    # with agreement), the shanks of one multi-shank session joined, so the
    # within-session drop the statistic tests can be seen session by session
    q = follx.dropna(subset=['excess']).assign(_sh=lambda d: (d.ml / 50).round() * 50)
    sh = (q.groupby(['mouse', 'day', '_sh'])
          .agg(n=('excess', 'size'), y=('excess', 'mean'), x=('ml', 'mean')).reset_index())
    sh = sh[sh.n >= MIN_SHANK_SP]
    n_multi = 0
    for _, g in sh.groupby(['mouse', 'day']):
        g = g.sort_values('x')
        if len(g) > 1 and g.x.max() - g.x.min() >= MIN_SPAN_UM:
            ax.plot(g.x, g.y, color='0.75', lw=.5, zorder=1)
            n_multi += 1
    ax.scatter(sh.x, sh.y, s=8, color='0.5', alpha=.7, lw=0, zorder=2)
    print(f'  {letter}: {len(sh)} shank points, {n_multi} multi-shank sessions joined')
    ax.set_xlim(2980, 3820)
    ax.set_xticks([3000, 3200, 3400, 3600, 3800])
    ax.axhline(0, color='0.6', lw=.7, ls=':')
    _rp = spearmanr(follx.dropna(subset=['excess']).ml, follx.dropna(subset=['excess']).excess)
    _ws, _p = within_session(follx, 'ml', 'excess')
    _b, _pm, _nmo, _nse, _fit = mixed_ml(follx, 'excess')
    # line, the mixed model's within-session slope (mouse random effect), as in D
    draw_mixed(ax, _fit, sh.x.min(), sh.x.max(), 'k', band_alpha=.2)
    ax.set_xlabel('medial–lateral (µm)', fontsize=8)
    ax.set_ylabel('agreement above chance', fontsize=8)
    _lp(ax, letter)
    ax.set_title(f'MEC cells, all sessions: pooled rho = {_rp[0]:+.2f} (p = {_rp[1]:.0e})\n'
                 f'within session, mouse as random effect: {_b:+.3f} per 100 µm, p = {_pm:.2g}',
                 fontsize=6.6, loc='left')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)
    print(f'  {letter} (ML): pooled rho {_rp[0]:+.3f} p {_rp[1]:.2g}; within {np.median(_ws):+.3f} '
          f'p {_p:.3g} n {len(_ws)}; MIXED (mouse) slope {_b:+.4f} per 100 um p {_pm:.3g} '
          f'{_nmo} mice {_nse} sessions')


def panel_border(ax, letter):
    """Distance from the MEC-parasubiculum border. Probe placement differs a lot
    between animals, which is why the pooled mediolateral coordinate is unsafe;
    distance from an anatomical landmark aligns every session to the same one.
    The border is taken from each animal's best-fit slice through the atlas:
    every parasubiculum pixel with an MEC neighbour is a border point, and a
    cell's distance is the SHORTEST distance from its projected position to any
    border point, signed by the cell's own atlas label (MEC +, parasubiculum -).
    Cells are at per-shank nominal mediolateral positions, so this is no finer
    than the shank pitch."""
    from scipy.spatial import cKDTree
    CB = (C.merge(_EX, on=['mouse', 'day', 'cluster_id'], how='left')
          .dropna(subset=['coord_SCs_x', 'coord_SCs_y', 'coord_SCs_z']))
    CB = CB[CB.brain_region.astype(str).str.match(r'^(ENTm|PAR|PRE)')].copy()
    CB['dist'] = np.nan
    for _mo, _g in CB.groupby('mouse'):
        try:
            _pts = np.array([stereo_to_ccf(r_) for r_ in
                             _g[['coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x']].values])
            _fit = fit_best_slice(get_mouse_contacts(_mo), extent_pts=_pts)
            _uv, _vv, _grid = sample_region_grid(_fit)
        except Exception as _e:
            print(f'  border: M{_mo} skipped ({type(_e).__name__})'); continue
        _uv, _vv = np.asarray(_uv), np.asarray(_vv)
        assert _grid.shape == (len(_vv), len(_uv)), (_grid.shape, len(_vv), len(_uv))
        _par, _mc = _grid == 2, np.isin(_grid, (3, 4))
        _nb = np.zeros_like(_par)
        _nb[1:, :] |= _mc[:-1, :]; _nb[:-1, :] |= _mc[1:, :]
        _nb[:, 1:] |= _mc[:, :-1]; _nb[:, :-1] |= _mc[:, 1:]
        _bi, _bj = np.where(_par & _nb)
        # fewer than ~20 border pixels means the slice only grazes the boundary
        # (M20 gives 4), which is too little to measure distance from
        if len(_bi) < 20:
            print(f'  border: M{_mo} has too little MEC-parasubiculum border in its slice '
                  f'({len(_bi)} points), skipped'); continue
        _tree = cKDTree(np.c_[_uv[_bj], _vv[_bi]])
        _pu, _pv = _fit['project'](_pts)
        _d, _ = _tree.query(np.c_[_pu, _pv])
        _sg = np.where(_g.brain_region.astype(str).str.startswith('ENTm'), 1.0, -1.0)
        CB.loc[_g.index, 'dist'] = _sg * _d
        print(f'  border: M{_mo} {len(_bi)} border points, {len(_g)} cells')
    CBv = CB.dropna(subset=['dist', 'excess'])
    for side, col in ((CBv.dist < 0, '0.55'), (CBv.dist >= 0, 'k')):
        x, y, e = binned(CBv[side], 'dist', 'excess', nbin=6 if col == '0.55' else 8)
        ax.fill_between(x, y - e, y + e, color=col, alpha=.25, lw=0, edgecolor='none')
        ax.plot(x, y, color=col, lw=1.8)
    ax.axvline(0, color='0.3', lw=.8, ls='--')
    ax.axhline(0, color='0.6', lw=.7, ls=':')
    ax.annotate('parasubiculum', (.02, .97), xycoords='axes fraction', fontsize=6.5,
                color='0.45', va='top')
    ax.annotate('MEC', (.98, .97), xycoords='axes fraction', fontsize=6.5, ha='right', va='top')
    _m = CBv[CBv.dist >= 0]
    _rb = spearmanr(_m.dist, _m.excess)
    _wb, _pb = within_session(_m.assign(ml=_m.dist), 'ml', 'excess')
    _near = CBv[CBv.dist.abs() <= 250]
    _pn = mannwhitneyu(_near[_near.dist < 0].excess, _near[_near.dist >= 0].excess).pvalue
    ax.set_xlabel('distance from the MEC–parasubiculum border (µm)', fontsize=8)
    ax.set_ylabel('agreement above chance', fontsize=8)
    _lp(ax, letter)
    ax.set_title(f'MEC side, pooled: rho = {_rb[0]:+.2f} (p = {_rb[1]:.0e}); within session '
                 f'{np.median(_wb):+.3f}, p = {_pb:.3g}\nwithin 250 µm of the border: '
                 f'parasubiculum {_near[_near.dist < 0].excess.mean():.3f} vs MEC '
                 f'{_near[_near.dist >= 0].excess.mean():.3f} (p = {_pn:.2g})',
                 fontsize=6.4, loc='left')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)
    print(f'  {letter} (border): MEC side pooled rho {_rb[0]:+.3f} p {_rb[1]:.2g}; within '
          f'{np.median(_wb):+.3f} p {_pb:.3g} n {len(_wb)}; near border PAR '
          f'{_near[_near.dist < 0].excess.mean():.3f} (n={int((_near.dist < 0).sum())}) MEC '
          f'{_near[_near.dist >= 0].excess.mean():.3f} (n={int((_near.dist >= 0).sum())}) '
          f'p {_pn:.3g}; cells with distance: {len(CBv)}')


def panel_axes(ax, letter):
    """The within-session distributions side by side, which is the actual test."""
    rng = np.random.default_rng(0)
    SUMM = [('M–L', 'ml'), ('D–V', 'dv')]
    for i, (nm, axis) in enumerate(SUMM):
        ws, p = within_session(foll, axis, 'r')
        ax.scatter(i + rng.uniform(-.16, .16, len(ws)), ws, s=13, color='0.45',
                   alpha=.8, lw=0, zorder=3)
        ax.plot([i - .28, i + .28], [np.median(ws)] * 2, color='k', lw=2, zorder=4)
        ax.annotate(f'p = {p:.3g}' if p >= .001 else f'p = {p:.1e}',
                    (i, .98), xycoords=('data', 'axes fraction'), ha='center', va='top',
                    fontsize=6.2, color='#a33' if p < .05 else '0.35')
    ax.axhline(0, color='0.6', lw=.8, ls=':')
    ax.set_xticks(range(len(SUMM)))
    ax.set_xticklabels([s_[0] for s_ in SUMM], fontsize=7.5)
    ax.set_xlim(-.5, len(SUMM) - .5)
    ax.set_ylabel('within-session rho', fontsize=8)
    _lp(ax, letter)
    ax.set_title(f'only M–L survives\n(M–L: {N_ML_SESS} multi-shank sessions)', fontsize=7, loc='left')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)


# ── controls and extensions, computed by fig4_controls.py / fig4_factors.py ───
import json as _json
CTL = pd.read_csv(f'{PS}/fig4_controls_cells.csv')
CM = CTL[CTL.mec]
FAC = _json.load(open(f'{PS}/fig4_factors.json'))


def _ws(df, y, min_cells=20):
    out = {}
    for k, g in df.dropna(subset=['ml', y]).groupby(['mouse', 'day']):
        if len(g) >= min_cells and g.ml.nunique() > 1 and g[y].nunique() > 1:
            out[k] = spearmanr(g.ml, g[y])[0]
    return pd.Series(out)


def panel_lso(ax, letter):
    """The gradient against an axis built without the cell's own shank."""
    _lp(ax, letter)
    a_, b_ = _ws(CM, 'r'), _ws(CM, 'r_lso')
    k_ = a_.index.intersection(b_.index)
    for kk in k_:
        ax.plot([0, 1], [a_[kk], b_[kk]], color='0.75', lw=.6, zorder=1)
    for i, v in enumerate((a_[k_], b_[k_])):
        ax.scatter(np.full(len(v), i), v, s=9, color='0.35', lw=0, zorder=2)
        ax.plot([i - .22, i + .22], [np.median(v)] * 2, color='k', lw=2, zorder=3)
        pv_ = wilcoxon(v).pvalue
        ax.annotate(f'p = {pv_:.3f}', (i, .98), xycoords=('data', 'axes fraction'),
                    ha='center', va='top', fontsize=6.2, color='#a33' if pv_ < .05 else '0.35')
    ax.axhline(0, color='0.6', lw=.8, ls=':')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['all\nshanks', 'other\nshanks'], fontsize=7)
    ax.set_xlim(-.5, 1.5)
    ax.set_ylabel('within-session rho\n(position vs agreement)', fontsize=8)
    ax.set_title('PC1 built without the\ncell\'s own shank', fontsize=7, loc='left')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)


def panel_steps(ax, letter):
    """How much of the position effect survives the factors it covaries with.
    Lateral cells are less theta-modulated and carry less spatial information,
    so part of the gradient is shared with those."""
    _lp(ax, letter)
    _st = list(FAC['ml_steps'].items())
    for i, (nm, v) in enumerate(_st):
        ax.errorbar(v['b'], i, xerr=1.96 * v['se'], fmt='o', color='k' if v['p'] < .05 else '0.55',
                    ms=4, lw=1, capsize=0)
        ax.annotate(f'p = {v["p"]:.2g}', (1.0, i), xycoords=('axes fraction', 'data'),
                    ha='right', va='center', fontsize=6.2, color='#a33' if v['p'] < .05 else '0.35')
    ax.axvline(0, color='0.6', lw=.8, ls=':')
    # fig4_factors.py names the base model after the figure it was first drawn in
    _nm = {'Figure 4 model': 'identity, layer, rate'}
    ax.set_yticks(range(len(_st))); ax.set_yticklabels([_nm.get(k, k) for k, _ in _st], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel('mediolateral coefficient\n(per SD lateral, 95% CI)', fontsize=7.5)
    ax.set_title(f"mediolateral position in the mixed model (shanks in sessions\n"
                 f"in mice), as more factors are added", fontsize=7, loc='left')
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    _xl = ax.get_xlim(); ax.set_xlim(_xl[0], _xl[1] + .55 * (_xl[1] - _xl[0]))


def panel_forest(ax, letter):
    """Every factor in one model."""
    _lp(ax, letter, dx=-.42)
    _grp = {r['factor']: r['group'] for r in FAC['factors']}
    _ph = {r['factor']: r['p_holm'] for r in FAC['factors']}
    _TERMS = [('ml_z', 'ml', 'mediolateral'), ('dv_z', 'dv', 'dorsoventral'),
              ('sup', 'layer', 'superficial (vs deep)'),
              ('C(identity, Treatment("non-grid spatial"))[T.grid]', 'type', 'grid (vs non-grid spatial)'),
              ('C(identity, Treatment("non-grid spatial"))[T.non-spatial]', 'type', 'non-spatial'),
              ('C(identity, Treatment("non-grid spatial"))[T.putative interneuron]', 'type', 'interneuron'),
              ('rate_z', 'rate', 'firing rate (log)'), ('wave_z', 'wave', 'waveform width'),
              ('si_z', 'si', 'spatial information'), ('th_z', 'theta', 'theta modulation'),
              ('gs_z', 'grid', 'grid score'), ('bs_z', 'border', 'border score'),
              ('hd_z', 'hd', 'head-direction tuning'), ('spd_z', 'speed', 'speed tuning |r|')]
    _y, _ticks, _prev = 0, [], None
    for t, f_, lab in _TERMS:
        if _prev is not None and _grp[f_] != _prev:
            _y += .6
        c = FAC['coef'][t]
        col = 'k' if _ph[f_] < .05 else '0.55'
        ax.errorbar(c['b'], _y, xerr=1.96 * c['se'], fmt='o', color=col, ms=3.8, lw=1, capsize=0)
        _ticks.append((_y, lab)); _prev = _grp[f_]; _y += 1
    ax.axvline(0, color='0.6', lw=.8, ls=':')
    ax.set_yticks([a for a, _ in _ticks]); ax.set_yticklabels([b for _, b in _ticks], fontsize=6.5)
    ax.invert_yaxis()
    _g0 = {}
    for (yy, _), (t, f_, _) in zip(_ticks, _TERMS):
        _g0.setdefault(_grp[f_], []).append(yy)
    for g_, ys in _g0.items():
        ax.annotate(g_, (1.02, np.mean(ys)), xycoords=('axes fraction', 'data'), fontsize=6.5,
                    color='0.35', va='center', rotation=-90)
    ax.set_xlabel('coefficient on agreement above chance\n(per SD, or vs reference; 95% CI)', fontsize=7.5)
    ax.set_title('all factors in one model; black = factor significant\n(likelihood ratio, Holm)',
                 fontsize=7, loc='left')
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)


def panel_dominance(ax, letter):
    """Ranked by dominance: each factor's added R^2 averaged over every subset of
    the others, so variance that overlapping factors share (interneurons are
    classed on rate and waveform, grid cells on grid score) is divided between
    them rather than dropped, as a drop-one share would."""
    _lp(ax, letter, dx=-.36)
    # firing rate is held in every model as a covariate and not ranked: a cell's
    # labels come from its VR spikes, so rate sets how well agreement is measured
    # (fig4_factors.COVARIATES). Shares are of the variance beyond it.
    _F = pd.DataFrame(FAC['factors'])
    _F = (_F[~_F.factor.isin(FAC['covariates'])]
          .sort_values('dominance', ascending=False).reset_index(drop=True))
    _tot = FAC['r2_within'] - FAC['r2_covariates']
    _yy = np.arange(len(_F))
    ax.barh(_yy, 100 * _F.dominance / _tot, height=.62, color='0.3', lw=0)
    # 95% interval over a bootstrap of whole sessions (fig4_factors.dominance_boot)
    ax.errorbar(100 * _F.dominance / _tot, _yy,
                xerr=[100 * _F.dominance / _tot - _F.boot_lo, _F.boot_hi - 100 * _F.dominance / _tot],
                fmt='none', ecolor='0.55', lw=.9, capsize=0)
    for i, r_ in _F.iterrows():
        ax.annotate(('p < 1e-4' if r_.p_holm < 1e-4 else f'p = {r_.p_holm:.2g}'),
                    (r_.boot_hi, i), xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=6, color='#a33' if r_.p_holm < .05 else '0.45')
    ax.set_yticks(_yy); ax.set_yticklabels(_F.label, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel('share of explained variance (%, dominance analysis)', fontsize=7.5)
    ax.set_title(f"which factor matters, beyond firing rate (held in every model)\n"
                 f"the {len(_F)} explain {100 * _tot:.1f}% of within-session variance; rate alone "
                 f"{100 * FAC['r2_covariates']:.1f}%", fontsize=7, loc='left')
    ax.text(.98, .04, f'bars, 95% interval over {FAC["n_boot"]}\nbootstraps of sessions\n'
            'p: unique effect\n(likelihood ratio, Holm)', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=6, color='0.4')
    ax.set_xlim(0, _F.boot_hi.max() * 1.3)
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)


# ══ FIGURE 4: the responsive subnetwork is medial, where grid cells are ══════
# Merged 2026-11-01 from the old Figures 4 (where grid cells are) and 5 (where
# the following is). Kept: the OF1 rate maps, the grid gradient on the
# mediolateral axis and in layer, the two strongest examples (M25 D25, kept for
# continuity with the M25 rate maps; M26 D19, balanced states and 13
# transitions), drawn on the probe rather than on a best-fit slice, and the
# gradient's tests and controls. The rest is Figure S9 below.
fig = plt.figure(figsize=(14.2, 9.15))
# sized so each rate-map slot is square (0.48 in): any taller and the maps,
# drawn at equal aspect, float apart with gaps between the rows
G = fig.add_gridspec(3, 1, height_ratios=[2.0, .01, .8], hspace=.15, left=.07, right=.97)
# the two example sessions stacked beside the rate maps, so A reads straight
# into the sessions that show the same gradient in following
TOP = G[0].subgridspec(1, 2, width_ratios=[2.55, 1.9], wspace=.12)
LEFT = TOP[0].subgridspec(1, 2, width_ratios=[.2, 2.1], wspace=.02)
_sch = panel_ratemaps(LEFT[0], LEFT[1], 'A')
_bx = [example_row(_cell, _letter, _mo, _dy, merge_mec=True) for (_letter, _mo, _dy), _cell in
       zip([('B', 25, 25), ('C', 26, 19)], TOP[1].subgridspec(2, 1, hspace=.5))]
# the region key at the top right of B, beside its boxplot strip
_pb = _bx[0].get_position()
region_legend(None, simple=True, anchor=(_pb.x1 + .012, _pb.y1 + .012))
g3 = G[2].subgridspec(1, 2, wspace=.16, width_ratios=[4.2, 1.45])
_axD = panel_composition(g3[0], 'D')
# the schematic sits just left of the first column of rate maps; its slot is
# narrower than the drawing, so it is placed by hand (the arena box spans .2-.8
# of the axes width)
_ps, _pm = _sch.get_position(), LEFT[1].get_position(fig)
_w = 1.5 * _ps.width
_sch.set_position([_pm.x0 - .84 * _w, _ps.y0, _w, _ps.height])
panel_ml(fig.add_subplot(g3[1]), 'E')
# what predicts following (old F) is its own supplement, Figure S10, below
out = f'{FIG}/fig4_subnetwork_anatomy.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
plt.close(fig)

# ══ FIGURE S9: two more example sessions for Figure 4 ════════════════════════
# Cut on 2026-11-01 to the two examples alone, drawn as Figure 4B, C, with the
# region key at the top right of the first, as there.
fig = plt.figure(figsize=(14.2, 3.4))
G = fig.add_gridspec(1, 2, wspace=.3, left=.07, right=.9)
_bx = [example_row(_cell, _letter, _mo, _dy, merge_mec=True)
       for (_letter, _mo, _dy), _cell in zip([('A', 28, 25), ('B', 29, 25)], G)]
_pb = _bx[1].get_position()
region_legend(None, simple=True, anchor=(_pb.x1 + .012, _pb.y1 + .03))
out = f'{FIG}/fig4_supp_anatomy_more.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
plt.close(fig)

# ══ FIGURE S10: what predicts following (Figure 4F until 2026-11-01) ═════════
def panel_methods(ax, letter):
    """The ranking three ways: dominance on every cell, dominance on cells
    firing >= RATE_FLOOR Hz in the task, and permutation importance from
    gradient-boosted trees (no linearity assumed, whole sessions held out)."""
    _lp(ax, letter, dx=-.42)
    _F = pd.DataFrame(FAC['factors'])
    _F = (_F[~_F.factor.isin(FAC['covariates'])]
          .sort_values('dominance', ascending=False).reset_index(drop=True))
    _rf = FAC['rate_floor']
    sh = dict(all=100 * _F.dominance / (FAC['r2_within'] - FAC['r2_covariates']),
              floor=100 * _F.dominance_floor / (_rf['r2_within'] - _rf['r2_covariates']),
              trees=100 * _F.perm_importance.clip(lower=0) / _F.perm_importance.clip(lower=0).sum())
    sty = dict(all=('o', 'k', f'dominance, all {FAC["n_cells"]:,} cells'),
               floor=('s', '0.55', f'dominance, cells >= {_rf["hz"]:.0f} Hz ({_rf["n_cells"]:,})'),
               trees=('^', '#b8860b', 'boosted trees, permutation\n(whole sessions held out)'))
    yy = np.arange(len(_F))
    for j, (k, v) in enumerate(sh.items()):
        mk, col, lab = sty[k]
        ax.scatter(v, yy + (j - 1) * .22, marker=mk, s=14, color=col, lw=0, label=lab, zorder=3)
    for i in yy:
        ax.plot([min(sh[k][i] for k in sh), max(sh[k][i] for k in sh)], [i, i], color='0.85', lw=.8, zorder=1)
    ax.set_yticks(yy); ax.set_yticklabels(_F.label, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel('share of explained variance (%)', fontsize=7.5)
    ax.set_title('the ranking three ways', fontsize=7, loc='left')
    ax.legend(fontsize=5.6, frameon=False, loc='lower right', handletextpad=.3)
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    _rk = {k: v.rank(ascending=False).values for k, v in sh.items()}
    from scipy.stats import spearmanr as _sp
    print(f'  S10 ranking agreement: floor rho {_sp(_rk["all"], _rk["floor"])[0]:+.2f}, '
          f'trees rho {_sp(_rk["all"], _rk["trees"])[0]:+.2f}')
    print('  S10 bootstrap: ' + ', '.join(f'{r.label} median rank {r.boot_rank_med:.0f}, '
                                          f'top in {100 * r.boot_p_top:.0f}%' for _, r in _F.iterrows()))


def panel_varparts(ax, letter):
    """Where the variance in agreement above chance lies, from the full mixed
    model: mouse, session within mouse, shank within session, the twelve
    measured cell properties together, and what is left (cell-to-cell)."""
    _lp(ax, letter, dx=-.3)
    R = FAC['r2']
    parts = [('mouse', R['mouse'], '#6a51a3'), ('session', R['sess'], '#9e9ac8'),
             ('shank', R['shank'], '#cbc9e2'), ('12 cell\nproperties', R['r2m'], '0.25'),
             ('residual\n(cell to cell)', R['resid'], '0.88')]
    for i, (lab, v, col) in enumerate(parts):
        ax.bar(i, 100 * v, .7, color=col, lw=0)
        ax.text(i, 100 * v + 1, f'{100 * v:.1f}%', ha='center', va='bottom', fontsize=6.2)
    ax.set_xticks(range(len(parts))); ax.set_xticklabels([p_[0] for p_ in parts], fontsize=6.3)
    ax.set_ylabel('share of variance in\nagreement above chance (%)', fontsize=7.5)
    ax.set_title(f'where the variance lies: {FAC["n_mice"]} mice, {FAC["n_sess"]} sessions,\n'
                 f'{FAC["n_shanks"]} shanks, {FAC["n_cells"]:,} cells', fontsize=7, loc='left')
    ax.set_ylim(0, 100 * max(v for _, v, _ in parts) * 1.15)
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    print('  S10 variance: ' + ', '.join(f'{l.replace(chr(10), " ")} {100 * v:.1f}%' for l, v, _ in parts))


fig = plt.figure(figsize=(14.2, 8.2))
G = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=.42, left=.1, right=.97)
g1 = G[0].subgridspec(1, 2, wspace=.55, width_ratios=[1.3, 1])
g2 = G[1].subgridspec(1, 3, wspace=.75, width_ratios=[1.2, .9, 1.1])
print('\nFigure S10:')
panel_dominance(fig.add_subplot(g1[0]), 'A')
panel_methods(fig.add_subplot(g1[1]), 'B')
panel_forest(fig.add_subplot(g2[0]), 'C')
panel_varparts(fig.add_subplot(g2[1]), 'D')
panel_steps(fig.add_subplot(g2[2]), 'E')
out = f'{FIG}/fig4_supp_factors.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
plt.close(fig)

# ══ NUMBERS ONLY ═══════════════════════════════════════════════════════════
# The text quotes results whose panels are no longer drawn: the best-fit slices,
# grid fraction along the dorsoventral axis and by layer, distance from the
# MEC-parasubiculum border, every factor's coefficient, the per-axis
# within-session test, the other-shank control and the mixed-model steps. They
# are drawn here on a figure that is never saved, so the rebuild log still
# prints every number the text quotes.
fig = plt.figure(figsize=(14.2, 14.0))
G = fig.add_gridspec(3, 4, hspace=.5, wspace=.5)
print('\nnumbers only (not drawn in any figure):')
panel_grid_axis(fig.add_subplot(G[0, 0]), 'dv', 'dorsal–ventral (µm)', '-')
panel_layer(fig.add_subplot(G[0, 1]), '-')
panel_border(fig.add_subplot(G[0, 2]), '-')
panel_axes(fig.add_subplot(G[1, 1]), '-')
panel_lso(fig.add_subplot(G[1, 2]), '-')
plt.close(fig)

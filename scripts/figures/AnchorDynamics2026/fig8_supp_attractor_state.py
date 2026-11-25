"""Out of the paper since 2026-11-02 (merged into Figure S14, fig8_supp_attractor_more.py): the attractor bump in M25 D24.

Figure 9 and the simulation in Figure S13 both rest on one assumption that
neither of them tests: that the non-anchored state is an attractor which is
still running and merely uncorrected, rather than one that has come apart. The
two possibilities make opposite predictions about how concentrated the
module's activity is on the torus, and nothing else in the paper measures it.

The analysis is attractor_coherence.py; this script only draws it. It is a
supplement rather than a main figure because it settles an assumption rather
than adding a claim -- but it is the panel that decides whether the gated
attractor is a model of these recordings at all.

Writes fig8_supp_attractor_state.pdf
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import proj3d
from scipy.stats import wilcoxon

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
OUT = f'{FIG}/fig8_supp_attractor_state.pdf'

ANCH, NON = '#a8559e', '#3f9b8f'
TA = ListedColormap([NON, ANCH])
NORM = BoundaryNorm([-0.5, 0.5, 1.5], TA.N)

D = pd.read_csv(f'{PS}/attractor_coherence.csv')
D1 = pd.read_csv(f'{PS}/attractor_coherence_s1.csv')
D0 = pd.read_csv(f'{PS}/attractor_coherence_s0.01.csv')
E = np.load(f'{PS}/attractor_example.npz')
BT = pd.read_csv(f'{PS}/attractor_bump_by_trial.csv')
M = np.load(f'{FIG}/can_gate_model.npz')

fig = plt.figure(figsize=(9.6, 12.4))
gs = fig.add_gridspec(4, 20, hspace=.72, wspace=2.6, left=.055, right=.985,
                      top=.95, bottom=.05, height_ratios=[1.0, 4.3, 1.0, 1.0])


def sheet(phi):
    """The two phases carried onto the rhombic unit cell (Gardner et al.)."""
    return phi[0] - phi[1] / 2.0, np.sqrt(3) / 2.0 * phi[1]


def lp(ax, s, x=-.24, y=1.07):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def bare(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


def torus3d(fig, parent, cycle):
    """A torus with one of its two cycles traced in the phase colour wheel.

    Cycle 1 runs the long way round the hole and cycle 2 the short way through
    the tube: the two independent ways round the surface, which are the two
    phases B and C plot. The long cycle is drawn as a band on the surface so
    that it is hidden correctly where the body passes in front of it; the
    short cycle is drawn over the body: a ring through the tube is hidden
    almost entirely by the surface under this renderer's depth sorting, so the
    circle is taken on the torus surface itself, projected into two dimensions
    through the same camera, and drawn on top. The half whose outward normal
    faces the camera lies on the visible surface and is drawn solid; the half
    that passes behind the tube is drawn faint, so the cycle still reads as a
    complete loop around the tube.
    """
    bb = parent.get_position()
    w, h = bb.width * .62, bb.height * .34
    rect = [bb.x1 - w, bb.y1 + bb.height * .10, w, h]
    ax = fig.add_axes(rect, projection='3d')
    nu, nv = 160, 64
    u, v = np.linspace(0, 2 * np.pi, nu), np.linspace(0, 2 * np.pi, nv)
    U, V = np.meshgrid(u, v)
    R, r = 1.0, .36
    ax.plot_surface((R + r * np.cos(V)) * np.cos(U),
                    (R + r * np.cos(V)) * np.sin(U), r * np.sin(V),
                    color='0.82', shade=True, linewidth=0, edgecolor='none',
                    antialiased=False, rstride=1, cstride=1)
    if cycle == 1:
        rr = r * 1.02
        UU, VV = np.meshgrid(u, np.linspace(.30, .95, 10))
        C = plt.cm.hsv((UU % (2 * np.pi)) / (2 * np.pi))
        ax.plot_surface((R + rr * np.cos(VV)) * np.cos(UU),
                        (R + rr * np.cos(VV)) * np.sin(UU), rr * np.sin(VV),
                        facecolors=C, shade=False, linewidth=0,
                        edgecolor='none', antialiased=False, rstride=1,
                        cstride=1)
    ax.view_init(elev=44, azim=-62); ax.set_axis_off()
    ax.set_box_aspect((1, 1, .45))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(-.85, .85)
    if cycle == 2:
        fig.canvas.draw()
        u0, t = -2.1, np.linspace(0, 2 * np.pi, 240)
        X = (R + r * np.cos(t)) * np.cos(u0)
        Y = (R + r * np.cos(t)) * np.sin(u0)
        Z = r * np.sin(t)
        cam = np.array([np.cos(np.radians(44)) * np.cos(np.radians(-62)),
                        np.cos(np.radians(44)) * np.sin(np.radians(-62)),
                        np.sin(np.radians(44))])
        nrm = np.stack([np.cos(t) * np.cos(u0), np.cos(t) * np.sin(u0),
                        np.sin(t)], 1)
        vis = nrm @ cam > .02              # the half lying on the near surface
        pts = np.array([proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
                        for x, y, z in zip(X, Y, Z)])
        ov = fig.add_axes(rect); ov.set_xlim(0, 1); ov.set_ylim(0, 1)
        ov.axis('off'); ov.patch.set_alpha(0)
        f = ov.transAxes.inverted().transform(ax.transData.transform(pts))
        for k in range(len(t) - 1):
            on = vis[k] and vis[k + 1]
            ov.plot(f[k:k + 2, 0], f[k:k + 2, 1], solid_capstyle='round',
                    lw=2.3 if on else 1.3, alpha=1.0 if on else .30,
                    color=plt.cm.hsv(k / (len(t) - 1)))


def star(p):
    return (f'p = {p:.1e}'.replace('e-0', 'e-') if p < 1e-3
            else f'p = {p:.3f}')


NB = int(E['nbin'])
st = E['state'].astype(bool)
mo_, dy_ = int(E['mouse']), int(E['day'])
t0 = int(E['strip_t0'])
str_tr = E['strip_trial']
TA0, TA1 = int(str_tr[0]), int(str_tr[-1])

# ---- A: the population state that defines the two conditions ---------------
Z = np.load(f'{PS}/labels/M{mo_}D{dy_}.npz', allow_pickle=True)
C = pd.read_csv(f'{PS}/anchoring_cells.csv')
C = C[(C.mouse == mo_) & (C.day == dy_)]
sel = np.isin(Z['cluster_id'], C.cluster_id[C.in_population].values)
L, load = Z['labels'][sel], Z['pc1_load'][sel]
with np.errstate(invalid='ignore'):
    varies = np.array([np.isfinite(r).sum() > 1 and np.nanstd(r) > 0
                       for r in L])
L = L[np.where(varies)[0][np.argsort(load[varies])[::-1]]]
ntr = L.shape[1]


def mark(ax, w=.9):
    """Where panel D's trials come from."""
    for y in (TA0, TA1):
        ax.axhline(y, color='k', lw=w)


gA = gs[0, 0:5].subgridspec(1, 2, wspace=.12, width_ratios=[1, .30])
axA = fig.add_subplot(gA[0, 0])
lp(axA, 'A', x=-.30)
axA.imshow(L.T, aspect='auto', cmap=TA, norm=NORM, interpolation='nearest',
           extent=[0, L.shape[0], ntr, 0])
axA.set_xticks([]); axA.tick_params(labelsize=6.5)
axA.set_ylabel('trial', fontsize=7.5)
axA.set_xlabel(f'{L.shape[0]} MEC cells', fontsize=7.5)
axA.set_title(f'the population state\nM{mo_} D{dy_}', fontsize=7.6,
              loc='left', color='0.25')
axP = fig.add_subplot(gA[0, 1], sharey=axA)
pc1, tr = Z['pc1'], np.arange(ntr)
axP.fill_betweenx(tr, 0, pc1, where=pc1 >= 0, color=ANCH, lw=0,
                  edgecolor='none', interpolate=True)
axP.fill_betweenx(tr, 0, pc1, where=pc1 < 0, color=NON, lw=0,
                  edgecolor='none', interpolate=True)
axP.axvline(0, color='0.4', lw=.5)
axP.set_xticks([]); axP.tick_params(labelleft=False, length=0)
for sp in axP.spines.values():
    sp.set_visible(False)
axP.set_xlabel('PC1', fontsize=7.0)

# ---- B, C: the module's two torus phases, in trial order -------------------
# B and C are psi, the population's phase along one lattice axis each. Which
# two matters: on a LINEAR track the bump has one degree of freedom, so the
# track direction projects onto two of the three lattice axes almost equally
# and onto the third hardly at all. The two that advance are therefore nearly
# the same phase -- here they correlate at +0.96 and differ by 12 degrees on
# average -- and drawing both would be drawing one twice. B shows the axis
# that runs along the track and C the one that runs across it, which is the
# independent pair (circular r = +0.05).
AX = E['psi_all'].reshape(len(E['psi_all']), -1, NB)
_adv = np.array([abs(np.median([np.unwrap(r)[-1] - np.unwrap(r)[0]
                                for r in w])) for w in AX])
K_ALONG, K_ACROSS = int(np.argmax(_adv)), int(np.argmin(_adv))
PSI = np.stack([AX[K_ALONG], AX[K_ACROSS]])      # 2 x trials x bins
for k, (lab, pn, cols) in enumerate(
        ((r'$\psi_\parallel$', 'B', slice(6, 9)),
         (r'$\psi_\perp$', 'C', slice(9, 12)))):
    ax = fig.add_subplot(gs[0, cols])
    lp(ax, pn, x=-.09)
    ax.imshow(PSI[k], cmap='hsv', aspect='auto', interpolation='nearest',
              extent=[0, 200, ntr, 0], vmin=-np.pi, vmax=np.pi)
    ax.set_yticks([])
    ax.set_xlim(0, 200); ax.set_xticks([0, 100, 200])
    ax.set_xlabel('position (cm)', fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    if not k:
        ax.text(0, -.52, f'{int(E["n_grid"])} grid cells, one module',
                transform=ax.transAxes, fontsize=6.6, color='0.35')
    ax.set_title(f'{lab}  ' + ('along' if not k else 'across'),
                 fontsize=8.0, loc='left', color='0.25', pad=13)
    torus3d(fig, ax, k + 1)
    if k:                                   # one colour wheel for both maps
        bb = ax.get_position()
        cax = fig.add_axes([bb.x1 + .010, bb.y0, .007, bb.height])
        cb = fig.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(-np.pi, np.pi), cmap='hsv'), cax=cax)
        cb.set_ticks([-np.pi, 0, np.pi])
        cb.set_ticklabels(['$-\\pi$', '0', '$\\pi$'])
        cb.ax.tick_params(labelsize=6.0, length=2, pad=1.5)
        cb.outline.set_linewidth(.4)
        cb.set_label('population phase', fontsize=6.4, labelpad=1)

# ---- D: the module's grid cells in the open field ---------------------------
# The cells B and C are read from, as OF1 rate maps (computed as Figure 4A
# computes them, at the session's travel lag), ordered by the cell's phase on the
# along-track axis so that the phase progression of B is visible across them.
# Cached, so a rebuild does not need the raw open-field files.
OF_CACHE = f'{PS}/attractor_example_of1.npz'
Zc0 = np.load(f'{PS}/attractor_cache/M{mo_}D{dy_}.npz')
_mod = [int(c) for c in Zc0['module']]
if not os.path.exists(OF_CACHE):
    import pynapple as nap
    from spatial_manifolds.detect_grids import (_fill_nans_from_neighbors,
                                                compute_travel_projected,
                                                gaussian_filter_nan)
    stem = (f'/Volumes/datasetsSD/Ephys_data/clark2025/M{mo_}/D{dy_}/OF1/'
            f'sub-M{mo_}_ses-D{dy_}_typ-OF1')
    beh = nap.load_file(f'{stem}_beh.nwb')
    clu = nap.load_file(f'{stem}_srt-kilosort4_clusters.npz')
    # the travel lag is one value per session (cell_classifications.csv lists
    # it for only some of this session's cells, all with the same value)
    cls = pd.read_csv(f'{ROOT}/data/cell_classifications.csv')
    lag = float(cls[(cls.mouse == mo_) & (cls.day == dy_)].travel.median())
    pos = np.stack([beh['P_x'], beh['P_y']], axis=1)
    maps = []
    for cid in _mod:
        bl = compute_travel_projected(['P_x', 'P_y'], pos, pos, lag)
        tc = nap.compute_2d_tuning_curves(
            nap.TsGroup([clu[cid]]), np.stack([bl['P_x'], bl['P_y']], axis=1),
            nb_bins=(40, 40))[0][0]
        tc = gaussian_filter_nan(_fill_nans_from_neighbors(tc), sigma=(2.5, 2.5))
        maps.append(np.clip(tc, 0, np.nanpercentile(tc, 99)))
    np.savez(OF_CACHE, cells=np.array(_mod), maps=np.stack(maps))
_OF = np.load(OF_CACHE)
_Pm0 = Zc0['P'][[list(Zc0['cells']).index(c) for c in _mod]].astype(float)
_ad0 = [abs(np.median([np.unwrap(r)[-1] - np.unwrap(r)[0] for r in w.reshape(-1, NB)]))
        for w in np.stack([np.angle(np.exp(1j * q) @ _Pm0) for q in Zc0['phi_all']])]
_ph_along = Zc0['phi_all'][int(np.argmax(_ad0))]
_ord = np.argsort(_ph_along)
NC_OF = 7
NR_OF = int(np.ceil(len(_mod) / NC_OF))
_bx = fig.add_subplot(gs[0, 13:20]); _bb = _bx.get_position(); _bx.remove()
_W, _H = fig.get_size_inches()
_side = min(_bb.width * _W / NC_OF, _bb.height * _H / NR_OF) * .94   # inches, square
for i, j in enumerate(_ord):
    _x = _bb.x0 + (i % NC_OF) * _side / .94 / _W
    _y = _bb.y1 - (i // NC_OF + 1) * _side / .94 / _H
    ax = fig.add_axes([_x, _y, _side / _W, _side / _H])
    ax.imshow(_OF['maps'][j], cmap='viridis', origin='lower', interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if i == 0:
        lp(ax, 'D', x=-.25, y=1.12)
        ax.set_title(f'the {len(_mod)} grid cells of B and C in the open field (OF1),\n'
                     'ordered by phase along the track', fontsize=7.6, loc='left',
                     color='0.25', pad=4)


# ---- E: the bump's two coordinates through the session, and at the switch --
# Drawn as Gardner et al. draw it: the two sheet coordinates and the animal's
# position as continuous traces, a window showing that the bump's own speed
# follows the animal's, and the unit cell itself either side of the switch --
# tight while the state holds, scattered once it is lost.
GS1 = gs[1, :].subgridspec(2, 1, hspace=.30, height_ratios=[2.4, 2.5])
GT = GS1[0].subgridspec(4, 1, hspace=.22)
GB = GS1[1].subgridspec(4, 1, hspace=.8, height_ratios=[.9, 1, .9, 1])
_psit = E['psi_t']; _spd = E['spd_t']; _pos = E['pos_t']
_idx = E['idx_t']; _okt = E['ok_t'].astype(bool)
_t = np.arange(len(_spd)) * 0.1
_thc = np.stack(sheet(_psit)) / (2 * np.pi) % 1.0
NW, WIN = 9, 6                       # nine windows of six trials
starts = np.linspace(t0 - (NW // 2) * WIN, t0 + (NW - NW // 2 - 1) * WIN,
                     NW).astype(int)
_pc1t = pc1[np.clip(_idx, 0, len(pc1) - 1)]
# the two windows the panels below are taken from
# THE TWO SNIPPETS. The clearest stretch of each state: 48 consecutive running
# 100 ms bins inside one state, the one with the largest mean bump above its
# shuffle (the non-anchored one within NON_WIN, in the long non-anchored stretch
# late in the session). Chosen for clarity, so they illustrate the bump rather
# than measure it; G-J measure it.
_pvx, _shx = E['pv'], E['sh']
_aa, _nn = E['anch'].astype(bool), E['non'].astype(bool)
SNIP = 48
NON_WIN = (1340, 1460)               # s; the non-anchored snippet is taken near 1400 s


def _best(mask, win=None):
    good = np.convolve(mask.astype(float), np.ones(SNIP), 'valid') == SNIP
    sc = np.convolve(_pvx - _shx, np.ones(SNIP) / SNIP, 'valid')
    if win is not None:
        tt = np.arange(len(sc)) * 0.1
        good &= (tt >= win[0]) & (tt <= win[1])
    sc[~good] = -np.inf
    return int(np.argmax(sc))


SN = {'anchored': _best(_aa), 'non-anchored': _best(_nn, NON_WIN)}
_W = {k: (_t[i0], _t[i0 + SNIP - 1]) for k, i0 in SN.items()}
for r_, (v, col, lab) in enumerate((
        (_pc1t, '0.25', 'PC1'),
        (_thc[0], '#4c72b0', r'$\theta_1$'),
        (_thc[1], '#dd8452', r'$\theta_2$'),
        (_pos, 'k', 'pos\n(cm)'))):
    a = fig.add_subplot(GT[r_])
    if r_ == 0:
        a.fill_between(_t, 0, v, where=v >= 0, color=ANCH, lw=0,
                       interpolate=True, rasterized=True)
        a.fill_between(_t, 0, v, where=v < 0, color=NON, lw=0,
                       interpolate=True, rasterized=True)
        a.axhline(0, color='0.4', lw=.4)
        a.set_ylim(-abs(v).max() * 1.1, abs(v).max() * 1.1)
        a.set_yticks([])
    else:
        a.plot(_t, v, color=col, lw=.2, rasterized=True)
        a.set_ylim(0, 1 if r_ < 3 else 200)
        a.set_yticks([0, 1] if r_ < 3 else [0, 200])
    a.set_xlim(0, _t[-1])
    a.tick_params(labelsize=5.8, labelbottom=(r_ == 3))
    a.set_ylabel(lab, fontsize=6.6, labelpad=2)
    a.spines[['top', 'right']].set_visible(False)
    for (x0, x1), c_ in ((_W['anchored'], ANCH), (_W['non-anchored'], NON)):
        a.add_patch(plt.Rectangle((x0, a.get_ylim()[0]), x1 - x0,
                                  a.get_ylim()[1] - a.get_ylim()[0],
                                  fill=False, ec=c_, lw=1.1, zorder=6,
                                  clip_on=False if r_ == 3 else True))
    if r_ == 0:
        lp(a, 'E', x=-.055, y=1.10)
        a.set_title('the bump through the session; the boxes are the two snippets '
                    'below', fontsize=7.6, loc='left', color='0.25')
    if r_ == 3:
        a.set_xlabel('time (s)', fontsize=7.0)

# ---- the two snippets: the bump on the sheet, and its speed ----------------
# Each frame is the module's activity at one moment, laid out on the unit cell
# of the neural sheet by the cells' phases on the along- and across-track axes
# (the independent pair of B and C; the two axes that both advance along the
# track are nearly the same phase and would draw a stripe), kernel-smoothed
# between cells as sheet_pixels() does and scaled frame by frame. Every second
# 100 ms bin, so 24 frames span 4.8 s. Below each strip, the animal's speed and
# the bump's own, z-scored over the session's running bins, for 30 s around
# the snippet (shaded).
Zc = np.load(f'{PS}/attractor_cache/M{mo_}D{dy_}.npz')
_cl = list(Zc['cells'])
_mi = [_cl.index(c) for c in Zc['module']]
_Rm = Zc['R'][_mi].astype(float)
_pa = Zc['phi_all']
_Pm = Zc['P'][_mi].astype(float)
_ad = [abs(np.median([np.unwrap(r)[-1] - np.unwrap(r)[0] for r in w.reshape(-1, NB)]))
       for w in np.stack([np.angle(np.exp(1j * q) @ _Pm) for q in _pa])]
_PH = _pa[[int(np.argmax(_ad)), int(np.argmin(_ad))]]
_g = np.linspace(-np.pi, np.pi, 30, endpoint=False)
_GU, _GV = np.meshgrid(_g, _g, indexing='ij')
_e = np.linspace(-np.pi, np.pi, 31)
_EU, _EV = np.meshgrid(_e, _e, indexing='ij')
_X0, _Y0 = (_EU - _EV / 2) / (2 * np.pi), (np.sqrt(3) / 2 * _EV) / (2 * np.pi)
_d1 = (_PH[0][:, None, None] - _GU[None] + np.pi) % (2 * np.pi) - np.pi
_d2 = (_PH[1][:, None, None] - _GV[None] + np.pi) % (2 * np.pi) - np.pi
_K = np.exp(-(_d1 ** 2 + _d2 ** 2) / (2 * .62 ** 2))
_K /= _K.sum(0)
NF, STEP = 24, 2
_pp = E['psi'].reshape(2, -1, NB)
_dpdx = np.array([np.median(np.diff(np.unwrap(_pp[k][st], axis=1), axis=1))
                  / (200.0 / NB) for k in range(2)])
_bs = np.r_[np.nan, np.array([np.diff(np.unwrap(_psit[k])) / 0.1 / _dpdx[k]
                              for k in range(2)]).mean(0)]
_run = _okt & np.isfinite(_bs)
_zs = lambda v: (v - np.nanmean(v[_run])) / (np.nanstd(v[_run]) + 1e-9)
_ZA, _ZB = _zs(_spd), _zs(_bs)
for r_, (nm, col) in enumerate((('anchored', ANCH), ('non-anchored', NON))):
    i0 = SN[nm]
    ax = fig.add_subplot(GB[2 * r_])
    for k in range(NF):
        _F = (_K * _Rm[:, i0 + STEP * k][:, None, None]).sum(0)
        _F = (_F - _F.min()) / (_F.max() - _F.min() + 1e-9)
        ax.pcolormesh(_X0 + k * 1.62, _Y0, _F, cmap='magma', shading='flat',
                      rasterized=True)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_xlim(-.62, (NF - 1) * 1.62 + 1.0)
    ax.set_title(f'{nm}: the bump on the sheet, every 200 ms for 4.8 s '
                 f'(t = {_t[i0]:.0f} s)', fontsize=7.4, loc='left', color=col, pad=3)
    if not r_:
        lp(ax, 'F', x=-.025, y=1.0)
    av = fig.add_subplot(GB[2 * r_ + 1])
    c0 = _t[i0] + SNIP * .05
    w_ = (_t >= c0 - 15) & (_t <= c0 + 15)
    for v, cc, lab in ((_ZA, '0.55', 'the animal'), (_ZB, 'k', 'the bump')):
        vv = np.where(_okt, v, np.nan)
        av.plot(_t[w_] - c0, vv[w_], color=cc, lw=.9, label=lab)
    av.axvspan(_t[i0] - c0, _t[i0 + SNIP - 1] - c0, color=col, alpha=.18, lw=0)
    av.axhline(0, color='0.8', lw=.5, zorder=0)
    av.set_xlim(-15, 15); av.set_ylim(-3, 3.5)
    av.set_ylabel('speed (z)', fontsize=7.0)
    av.tick_params(labelsize=6.2)
    av.spines[['top', 'right']].set_visible(False)
    m_ = w_ & _run
    av.text(.99, .97, f'r = {np.corrcoef(_ZA[m_], _ZB[m_])[0, 1]:+.2f} over these 30 s',
            transform=av.transAxes, fontsize=5.8, ha='right', va='top', color='0.3')
    if not r_:
        av.tick_params(labelbottom=False)
        av.legend(fontsize=5.8, frameon=False, loc='upper left', ncol=2,
                  handlelength=1.0, columnspacing=.9, borderaxespad=.1)
    else:
        av.set_xlabel('time from the snippet (s)', fontsize=7.0)



# ---- F: how concentrated the population is, moment by moment ---------------
ax = fig.add_subplot(gs[2, 0:4])
lp(ax, 'G', x=-.30)
pv, sh = E['pv'], E['sh']
a, n_ = E['anch'].astype(bool), E['non'].astype(bool)
bins = np.linspace(min(pv.min(), sh.min()), max(pv.max(), sh.max()), 46)
for v, c, lab in ((sh[a | n_], '0.55', 'phase shuffled'),
                  (pv[n_], NON, 'non-anchored'), (pv[a], ANCH, 'anchored')):
    ax.hist(v, bins=bins, density=True, histtype='step', color=c, lw=1.2,
            label=lab)
bare(ax)
ax.set_xlabel('bump population vector, per 100 ms', fontsize=7.5)
ax.set_ylabel('density', fontsize=7.5)
ax.legend(fontsize=6.0, frameon=False, loc='upper right', handlelength=1.1,
          borderpad=.1, labelspacing=.25)
ax.set_title('both states above the shuffle', fontsize=7.6, loc='left',
             color='0.25')

# ---- F: every session, against its own shuffle -----------------------------
ax = fig.add_subplot(gs[2, 5:9])
lp(ax, 'H')
for _, r in D.iterrows():
    ax.plot([0, 1], [r.bump_anch, r.bump_non], color='0.78', lw=.6, zorder=1)
rng = np.random.default_rng(0)
MIN_RELIABLE = 15          # grid cells in the module; see K and L
_big = D.n_grid.values >= MIN_RELIABLE
for x, col, c in ((0, 'bump_anch', ANCH), (1, 'bump_non', NON)):
    v = D[col].values
    jx = np.full(len(v), x) + rng.normal(0, .03, len(v))
    ax.scatter(jx[_big], v[_big], s=9, color=c, lw=0, zorder=2)
    ax.scatter(jx[~_big], v[~_big], s=9, facecolor='none', edgecolor=c, lw=.6, zorder=2)
    ax.plot([x - .18, x + .18], [v.mean()] * 2, color=c, lw=1.8, zorder=3)
ax.axhline(0, color='k', lw=.8, ls=(0, (3, 2)))
ax.set_xlim(-.45, 1.45); ax.set_xticks([0, 1])
ax.set_xticklabels(['anchored', 'non-\nanchored'], fontsize=6.8)
ax.set_ylabel('bump above shuffle', fontsize=7.5)
bare(ax)
pa, pn_ = wilcoxon(D.bump_anch).pvalue, wilcoxon(D.bump_non).pvalue
pd_ = wilcoxon(D.bump_anch - D.bump_non).pvalue
ax.set_ylim(-.004, D.bump_anch.max() * 1.45)
psp = wilcoxon(D.sm_anch - D.sm_non).pvalue
ax.text(.5, .99, f'both above zero: {star(max(pa, pn_))}\n'
        f'anchored higher: {star(pd_)}\n'
        f'speeds equated: {star(psp)}', transform=ax.transAxes,
        fontsize=6.3, ha='center', va='top', color='0.25')
ax.set_title(f'{len(D)} sessions, {int(D.n_grid.sum())} grid cells',
             fontsize=7.6, loc='left', color='0.25')

# ---- G: the bump across the switch, trial by trial -------------------------
WLAG = 8


def _epochs(a):
    """State epochs, with runs shorter than three trials absorbed into the
    preceding one -- the definition of a transition used everywhere else in
    the paper, since a trial or two across the axis is a crossing rather than
    a state the population entered."""
    a, out, i = a.astype(int).copy(), [], 0
    while i < len(a):
        j = i
        while j < len(a) and a[j] == a[i]:
            j += 1
        if out and j - i < 3:
            a[i:j] = a[i - 1]
            out[-1] = (out[-1][0], j, out[-1][2])
        else:
            out.append((i, j, bool(a[i])))
        i = j
    return out


segs, slopes, dist = [], [], []
for (mo_s, dy_s), g in BT.groupby(['mouse', 'day']):
    g = g.sort_values('trial')
    tr, bb, aa = g.trial.values, g.bump.values, g.anch.values
    full = np.full(tr.max() + 1, np.nan); full[tr] = bb
    fa = np.zeros(tr.max() + 1, bool); fa[tr] = aa
    ep = _epochs(fa)
    for i, j, state in ep:
        if j - i >= 8:
            seg = full[i:j]; kk = np.arange(j - i); o = np.isfinite(seg)
            if o.sum() >= 6:
                slopes.append((state, np.polyfit(kk[o], seg[o], 1)[0]))
    for i, j, state in ep[1:]:
        if state or not fa[i - 1]:              # anchored -> non-anchored only
            continue
        if i - WLAG < 0 or i + WLAG > len(full):
            continue
        q = full[i - WLAG:i + WLAG]
        if np.isfinite(q).sum() >= 1.4 * WLAG:
            segs.append(q)
    cl = np.concatenate([np.full(j - i, st_) for i, j, st_ in ep])
    sw = np.flatnonzero(np.diff(cl.astype(int)) != 0) + .5
    d = (np.min(np.abs(np.arange(len(fa))[:, None] - sw[None, :]), 1)
         if len(sw) else np.full(len(fa), np.inf))
    for t, v in zip(tr, bb):
        dist.append((mo_s, dy_s, d[t], bool(cl[t]), v))
SEG = np.array(segs)
D2 = pd.DataFrame(dist, columns=['mouse', 'day', 'dist', 'anch', 'bump'])

ax = fig.add_subplot(gs[2, 10:14])
lp(ax, 'I')
lags = np.arange(-WLAG, WLAG)
m_ = np.nanmean(SEG, 0)
se_ = np.nanstd(SEG, 0) / np.sqrt(np.sum(np.isfinite(SEG), 0))
for sl_, c in ((slice(0, WLAG), ANCH), (slice(WLAG, 2 * WLAG), NON)):
    ax.plot(lags[sl_], m_[sl_], color=c, lw=1.3, marker='o', ms=2.6, mew=0)
    ax.fill_between(lags[sl_], (m_ - se_)[sl_], (m_ + se_)[sl_], color=c,
                    alpha=.22, linewidth=0, edgecolor='none')
ax.axvline(-.5, color='k', lw=.8, ls=(0, (3, 2)))
ax.set_xlabel('trial from the switch', fontsize=7.5)
ax.set_ylabel('bump above shuffle', fontsize=7.5)
step = np.nanmean(SEG[:, WLAG:WLAG + 3], 1) - np.nanmean(
    SEG[:, WLAG - 3:WLAG], 1)
p_step = wilcoxon(step[np.isfinite(step)]).pvalue
sl_non = np.array([v for s_, v in slopes if not s_])
p_slope = wilcoxon(sl_non).pvalue
bare(ax)
ax.set_ylim(.018, .072)
ax.text(.5, .99, f'no step: {star(p_step)}\nno decay after: {star(p_slope)}',
        transform=ax.transAxes, fontsize=6.2, ha='center', va='top',
        color='0.25')
ax.set_title(f'{len(SEG)} switches: it does not\ndecay into the new state',
             fontsize=7.6, loc='left', color='0.25')

# ---- H2: how far from the switch the difference is expressed ---------------
BINS = ((0, 2, '0-2'), (3, 5, '3-5'), (6, 10, '6-10'), (11, 1e9, '>10'))
mm, ee, pp = [], [], []
for lo, hi, _ in BINS:
    q = D2[(D2.dist >= lo) & (D2.dist <= hi)]
    S_ = q.groupby(['mouse', 'day']).apply(
        lambda g: pd.Series({'a': g[g.anch].bump.mean(),
                             'n': g[~g.anch].bump.mean()}),
        include_groups=False).dropna()
    d_ = (S_.a - S_.n).values
    mm.append(d_.mean()); ee.append(d_.std() / np.sqrt(len(d_)))
    pp.append(wilcoxon(d_).pvalue)
# the smoothing series and the model comparison are reported in the Methods
RET = np.stack([100 * T.bump_non.values / T.bump_anch.values
                for T in (D0, D1, D)])
keep = np.median(RET, axis=1)
mo, mn = M['bump_open'], M['bump_shut']
mk = 100 * (mn[0] - mn[1]) / (mo[0] - mo[1])
p_sm = wilcoxon(RET[0] - RET[2]).pvalue

# ---- I: the bump's speed against the animal's, every session ---------------
# attractor_coherence.bump_speed: Pearson r over consecutive running 100 ms
# bins (the same as r between the two z-scored speeds), against a phase
# shuffle that keeps every rate; and the same with the two states' speed
# distributions matched, since a narrower speed range lowers r by itself.
import statsmodels.formula.api as smf
SP = pd.read_csv(f'{PS}/attractor_speed.csv')
ax = fig.add_subplot(gs[2, 15:20])
lp(ax, 'J', x=-.30)
for _, r in SP.iterrows():
    ax.plot([0, 1], [r.r_a, r.r_n], color='0.78', lw=.6, zorder=1)
SP = SP.merge(D[['mouse', 'day', 'n_grid']], on=['mouse', 'day'])
_bg = SP.n_grid.values >= MIN_RELIABLE
for x_, v, c in ((0, SP.r_a.values, ANCH), (1, SP.r_n.values, NON)):
    jx = np.full(len(v), x_) + rng.uniform(-.06, .06, len(v))
    ax.scatter(jx[_bg], v[_bg], s=9, color=c, lw=0, zorder=2)
    ax.scatter(jx[~_bg], v[~_bg], s=9, facecolor='none', edgecolor=c, lw=.6, zorder=2)
    ax.plot([x_ - .2, x_ + .2], [np.median(v)] * 2, color=c, lw=2, zorder=3)
_nul = np.r_[SP.rnull_a, SP.rnull_n].mean()
ax.axhline(_nul, color='0.3', lw=.8, ls=(0, (3, 2)))
ax.text(1.42, _nul, 'shuffle', fontsize=5.6, color='0.3', va='center', ha='left')
p_ea = wilcoxon(SP.r_a - SP.rnull_a).pvalue
p_en = wilcoxon(SP.r_n - SP.rnull_n).pvalue
p_d = wilcoxon(SP.r_a, SP.r_n).pvalue
p_dm = smf.mixedlm('d ~ 1', SP.assign(d=SP.r_a - SP.r_n),
                   groups=SP['mouse']).fit(reml=True).pvalues['Intercept']
p_m = wilcoxon(SP.rm_a, SP.rm_n).pvalue
p_sl = wilcoxon(SP.slope_a, SP.slope_n).pvalue
ax.set_xticks([0, 1]); ax.set_xticklabels(['anchored', 'non-\nanchored'], fontsize=6.5)
ax.set_xlim(-.45, 1.45)
ax.set_ylabel('r, bump speed with\nthe animal\'s speed', fontsize=7.5)
ax.text(.5, .99, f'both above shuffle: {star(max(p_ea, p_en))}\n'
        f'anchored higher: {star(p_d)}\nspeeds matched: {star(p_m)}',
        transform=ax.transAxes, fontsize=6.0, ha='center', va='top', color='0.25')
ax.set_ylim(min(-.01, SP.r_n.min() - .01), SP.r_a.max() * 1.45)
bare(ax)
ax.set_title(f'{len(SP)} sessions: the bump keeps\npace, less closely when lost',
             fontsize=7.6, loc='left', color='0.25')
print(f'  I bump-animal speed r: anchored {SP.r_a.median():.3f} (above shuffle '
      f'{(SP.r_a > SP.rnull_a).sum()}/{len(SP)}, p={p_ea:.2g}), non-anchored '
      f'{SP.r_n.median():.3f} ({(SP.r_n > SP.rnull_n).sum()}/{len(SP)}, p={p_en:.2g}); '
      f'difference p={p_d:.2g} (mixed, mouse p={p_dm:.2g}), anchored higher in '
      f'{(SP.r_a > SP.r_n).sum()}/{len(SP)}; speed-matched {SP.rm_a.median():.3f} vs '
      f'{SP.rm_n.median():.3f} p={p_m:.2g}; slope {SP.slope_a.median():.2f} vs '
      f'{SP.slope_n.median():.2f} cm per cm, p={p_sl:.2g}; speed SD '
      f'{SP.spd_sd_a.median():.1f} vs {SP.spd_sd_n.median():.1f} cm/s; '
      f'{SP.mouse.nunique()} mice')

# ---- K: how many grid cells the bump measure needs -------------------------
# attractor_reliability.py: the module split into two disjoint random halves of
# k cells, the bump's phase read from each, and their moment-to-moment
# agreement (mean cos of the difference, both axes, both states; 100 splits).
RL = pd.read_csv(f'{PS}/attractor_reliability.csv')
AG = RL.dropna(subset=['agree_a']).assign(ag=lambda d: (d.agree_a + d.agree_n) / 2)
ax = fig.add_subplot(gs[3, 0:9])
lp(ax, 'K', x=-.13)
for (m_, d_), g in AG.groupby(['mouse', 'day']):
    big = g.n_grid.iloc[0] >= MIN_RELIABLE
    ax.plot(g.k, g.ag, color='0.2' if big else '0.72', lw=.9 if big else .8, zorder=2 if big else 1)
    ax.plot(g.k.iloc[-1], g.ag.iloc[-1], 'o', ms=2.6, color='0.2' if big else '0.72', zorder=3)
ax.axhline(0, color='k', lw=.6, ls=(0, (3, 2)))
ax.set_xlabel('grid cells in each half', fontsize=7.5)
ax.set_ylabel('split-half agreement\nof the bump\'s phase', fontsize=7.5)
ax.set_xlim(1.5, AG.k.max() + .5)
bare(ax)
k4 = AG[AG.k == 4].groupby(['mouse', 'day']).agg(ag=('ag', 'first'), n=('n_grid', 'first'))
from scipy.stats import spearmanr
rho4, prho4 = spearmanr(k4.n, k4.ag)
ax.set_title(f'agreement grows with the cells per half; at a matched 4 per half,\n'
             f'sessions with small modules are no worse (rho = {rho4:+.2f}, p = {prho4:.2f})',
             fontsize=7.2, loc='left', color='0.25')
ax.text(.99, .05, f'dark: modules of at least {MIN_RELIABLE} grid cells; dot, half the module',
        transform=ax.transAxes, fontsize=5.8, ha='right', color='0.35')

# ---- L: the measures against module size -----------------------------------
ax = fig.add_subplot(gs[3, 11:20])
lp(ax, 'L', x=-.13)
SB = AG.groupby(['mouse', 'day']).apply(lambda g: g.ag.iloc[-1], include_groups=False)
SB = (2 * SB / (1 + SB)).rename('sb').reset_index().merge(D[['mouse', 'day', 'n_grid']])
ax.scatter(SB.n_grid, SB.sb, s=14, c=['0.2' if n >= MIN_RELIABLE else '0.72' for n in SB.n_grid],
           lw=0, zorder=3)
ax.axvline(MIN_RELIABLE - .5, color='k', lw=.7, ls=(0, (3, 2)))
ax.set_xlabel('grid cells in the module', fontsize=7.5)
ax.set_ylabel('reliability of the\nwhole module (est.)', fontsize=7.5)
ax.set_ylim(0, .7)
bare(ax)
Db = D[D.n_grid >= MIN_RELIABLE]
Sb = SP[SP.n_grid >= MIN_RELIABLE]
p_b = wilcoxon(Db.bump_anch - Db.bump_non).pvalue
p_r = smf.mixedlm('d ~ 1', Sb.assign(d=Sb.r_a - Sb.r_n), groups=Sb['mouse']).fit(
    reml=True).pvalues['Intercept']
ax.set_title(f'{len(Db)} sessions ({Db.mouse.nunique()} mice) have at least {MIN_RELIABLE} grid cells. '
             f'In them alone: both states\nabove shuffle {star(max(wilcoxon(Db.bump_anch).pvalue, wilcoxon(Db.bump_non).pvalue))}; '
             f'anchored higher {star(p_b)}; speed r higher\n{star(p_r)} (mouse random effect)',
             fontsize=7.0, loc='left', color='0.25')
print('  sessions by module size: ' + ', '.join(f'M{r.mouse}D{r.day} {r.n_grid}'
                                                for r in D.sort_values('n_grid').itertuples()))
print(f'  >= {MIN_RELIABLE} grid cells: {len(Db)} sessions, {Db.mouse.nunique()} mice; bump anchored '
      f'{Db.bump_anch.mean():+.4f}, non {Db.bump_non.mean():+.4f}, difference p = {p_b:.2g}; speed r '
      f'{Sb.r_a.median():.3f} vs {Sb.r_n.median():.3f}, mixed p = {p_r:.2g}; K rho at k=4 {rho4:+.2f} p {prho4:.2g}')

fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'wrote {OUT}')
print(f'  model keeps {mk:.0f}%; recordings {keep[0]:.0f}/{keep[1]:.0f}/'
      f'{keep[2]:.0f}% at none/100/250 ms, none vs 250 p = {p_sm:.3f}')
print(f'  anchored {D.bump_anch.mean():+.4f} p={pa:.2g}; non-anchored '
      f'{D.bump_non.mean():+.4f} p={pn_:.2g}; difference p={pd_:.2g}')
print(f'  {len(SEG)} switches: step {np.nanmean(step):+.4f} p={p_step:.3g}; '
      f'non-anchored within-epoch slope {sl_non.mean():+.5f} p={p_slope:.3g}')
print('  difference by distance from switch: ' +
      ', '.join(f'{b[2]} {v:+.4f} (p={q:.3g})'
                for b, v, q in zip(BINS, mm, pp)))

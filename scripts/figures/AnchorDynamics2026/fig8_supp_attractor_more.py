"""The attractor bump survives while non-anchored. With --sim this builds main Figure 9
(A: the model's schematic; B: a simulated gated-attractor module; C: M29 D25; D-F);
without it, the two-session version with reliability (G, H) and one-bump (I-L) panels,
out of the paper since 2026-11-02.

The model of Figure 10 assumes the non-anchored state is an attractor that is
still running and merely uncorrected, rather than one that has come apart.
This figure tests it. Each grid cell gets phase offsets on its module's torus
from the spectral method, the population is projected onto them, and every
result is referred to a shuffle of which offset belongs to which cell.

B, C  a simulated and a recorded module: where the mouse stopped on each trial
      (recorded: uncued trials only), the population state (cell by cell and
      PC1; triangles, the trials drawn below), the bump's phase along and
      across the track at each position of each trial (with the torus cycle
      each one runs round, in B), the module in the open field ordered by
      phase along the track with where its cells sit on the sheet below, the
      bump on the sheet over three consecutive trials in each state with the
      animal's and the bump's speed, and what the bump says about position
      and speed on those trials
D-F   the bump in every session (attractor_coherence.py): moment by moment in
      the examples, against its shuffle in every session, and its speed
      against the animal's; the bump across the switch is printed, not drawn
G, H  how many grid cells the measure needs (attractor_reliability.py)
I-L   one bump or two: the extra blobs of non-anchored frames
      (attractor_multibump.py)

Writes fig8_supp_attractor_more.pdf; with --sim (or SIM=1), Figure 9:
fig8_supp_attractor_sim.pdf
"""
import os
import sys

import matplotlib
import matplotlib.transforms
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attractor_speed_cycle as SC                               # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import proj3d
from scipy.stats import spearmanr, wilcoxon

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
# SIM=1 builds Figure S15: the same figure with B a simulated session of the
# gated attractor (attractor_sim_session.py) instead of M25 D25
SIM = os.environ.get('SIM') == '1' or '--sim' in sys.argv
OUT = f'{FIG}/fig8_supp_attractor_sim.pdf' if SIM else f'{FIG}/fig8_supp_attractor_more.pdf'
ANCH, NON = '#a8559e', '#3f9b8f'
TA = ListedColormap([NON, ANCH])
NORM = BoundaryNorm([-0.5, 0.5, 1.5], TA.N)
# (0, 0) is the simulation; in Figure 9 it comes first, beside a schematic of the model
EXAMPLES = [(0, 0), (29, 25)] if SIM else [(29, 25), (25, 25)]
# where the laps are taken from, by session and state (anchored True), as trial
# rows of the state raster; chosen by eye from the raster and PC1
WIN = {(29, 25, False): (150, 10 ** 6), (25, 25, False): (120, 156)}
# or a run given outright: its first trial row
FIX = {(25, 25, False): 127}             # trials 140-142, mid-epoch
MIN_RELIABLE = 15                        # grid cells in the module; G, H
TL, NPERM = 200.0, 10
RZ, STOP_SPD = (90.0, 110.0), 3.0       # reward zone (cm); the task's stop criterion (cm/s)
POS = list(range(5, 200, 10))            # cm, one frame each
SP, RS = 1.04, .93                       # frame spacing, lap spacing (sheet units)
TT = pd.read_csv(f'{PS}/attractor_trial_types.csv')
rng = np.random.default_rng(0)


def lp(ax, s, x=-.24, y=1.07):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom', ha='right')


def bare(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


def star(p):
    return f'p = {p:.1e}'.replace('e-0', 'e-') if p < 1e-3 else f'p = {p:.2g}'


def mixed(y):
    """Mean and p of a per-session quantity, mouse as a random effect."""
    d = y.dropna().reset_index()
    d.columns = ['mouse', 'day', 'y']
    r = smf.mixedlm('y ~ 1', d, groups=d.mouse).fit(reml=True)
    return r.params.iloc[0], r.pvalues.iloc[0]


def of_maps(mo, dy, cells):
    """OF1 rate maps of the module's cells (as Figure 4A computes them); cached."""
    f = f'{PS}/attractor_of1_M{mo}D{dy}.npz'
    if not os.path.exists(f):
        import pynapple as nap
        from spatial_manifolds.detect_grids import (_fill_nans_from_neighbors,
                                                    compute_travel_projected,
                                                    gaussian_filter_nan)
        stem = f'/Volumes/datasetsSD/Ephys_data/clark2025/M{mo}/D{dy}/OF1/sub-M{mo}_ses-D{dy}_typ-OF1'
        beh = nap.load_file(f'{stem}_beh.nwb')
        clu = nap.load_file(f'{stem}_srt-kilosort4_clusters.npz')
        cls = pd.read_csv(f'{ROOT}/data/cell_classifications.csv')
        lag = float(cls[(cls.mouse == mo) & (cls.day == dy)].travel.median())
        pos = np.stack([beh['P_x'], beh['P_y']], axis=1)
        bl = compute_travel_projected(['P_x', 'P_y'], pos, pos, lag)
        pl = np.stack([bl['P_x'], bl['P_y']], axis=1)
        maps = []
        for cid in cells:
            tc = nap.compute_2d_tuning_curves(nap.TsGroup([clu[cid]]), pl, nb_bins=(40, 40))[0][0]
            tc = gaussian_filter_nan(_fill_nans_from_neighbors(tc), sigma=(2.5, 2.5))
            maps.append(np.clip(tc, 0, np.nanpercentile(tc, 99)))
        np.savez(f, cells=np.array(cells), maps=np.stack(maps))
    return np.load(f)['maps']


def coherence(R, ph):
    n = np.linalg.norm(R, axis=0) + 1e-9
    return np.mean([np.abs(np.exp(1j * q) @ R) for q in ph], 0) / (np.sqrt(R.shape[0] / 2) * n)


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
    # One surface, with the cycle coloured into its own faces: two separate
    # surfaces are depth-sorted as wholes, which hides a band laid over the
    # body, but faces of one surface are sorted against each other. Cycle 1
    # runs the long way round the hole, cycle 2 the short way round the tube;
    # the body is shaded by hand, since facecolors switch shading off.
    nu, nv = 200, 90
    u, v = np.linspace(0, 2 * np.pi, nu), np.linspace(0, 2 * np.pi, nv)
    U, V = np.meshgrid(u, v)
    R, r = 1.0, .36
    nrm = np.stack([np.cos(V) * np.cos(U), np.cos(V) * np.sin(U), np.sin(V)], -1)
    lgt = np.array([-.35, -.55, .76]); lgt /= np.linalg.norm(lgt)
    shade = .62 + .38 * np.clip(nrm @ lgt, 0, 1)
    C = np.ones(U.shape + (4,))
    C[..., :3] = (.84 * shade)[..., None]
    if cycle == 1:
        band = (V > .30) & (V < .95)
        C[band] = plt.cm.hsv(U[band] / (2 * np.pi))
    else:
        u0 = (-2.1) % (2 * np.pi)
        band = np.abs((U - u0 + np.pi) % (2 * np.pi) - np.pi) < .2
        C[band] = plt.cm.hsv(V[band] / (2 * np.pi))
    C[band, :3] *= (.75 + .25 * shade[band])[:, None]
    ax.plot_surface((R + r * np.cos(V)) * np.cos(U), (R + r * np.cos(V)) * np.sin(U),
                    r * np.sin(V), facecolors=C, shade=False, linewidth=0, edgecolor='none',
                    antialiased=False, rstride=1, cstride=1)
    ax.view_init(elev=44, azim=-62); ax.set_axis_off()
    ax.set_box_aspect((1, 1, .45))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(-.85, .85)


_g = np.linspace(-np.pi, np.pi, 30, endpoint=False)
_GU, _GV = np.meshgrid(_g, _g, indexing='ij')
_e = np.linspace(-np.pi, np.pi, 31)
_EU, _EV = np.meshgrid(_e, _e, indexing='ij')
_X0, _Y0 = (_EU - _EV / 2) / (2 * np.pi), (np.sqrt(3) / 2 * _EV) / (2 * np.pi)


def kernel(ph):
    d1 = (ph[0][:, None, None] - _GU[None] + np.pi) % (2 * np.pi) - np.pi
    d2 = (ph[1][:, None, None] - _GV[None] + np.pi) % (2 * np.pi) - np.pi
    K = np.exp(-(d1 ** 2 + d2 ** 2) / (2 * .62 ** 2))
    return K / K.sum(0)


def schematic(ax):
    """The model as a flow, left to right, for Figure 9's top row: self-motion is
    always integrated by the attractor sheet; the landmark correction reaches it
    only through the attentional gate; the simulated mouse decodes its position
    from the bump and stops where that says the reward zone is; 36 units are
    sampled and analysed exactly as the recordings."""
    ax.set_xlim(0, 10); ax.set_ylim(-.15, 1.3); ax.set_aspect('equal'); ax.axis('off')
    box = dict(boxstyle='round,pad=0.3', fc='0.95', ec='0.55', lw=.5)
    arr = dict(arrowstyle='-|>', lw=.8, color='0.3', mutation_scale=7)
    fs = 6.2
    ax.text(.75, 1.0, 'self-motion\n(running speed)', fontsize=fs, ha='center', va='center', bbox=box)
    ax.annotate('', (2.05, .78), (1.45, .95), arrowprops=arr)
    g = np.linspace(-np.pi, np.pi, 41)
    U_, V_ = np.meshgrid(g, g, indexing='ij')
    B_ = np.exp(1.6 * (np.cos(U_ - .5) + np.cos(V_ + .4) + np.cos(U_ - .5 + V_ + .4) - 3))
    sc, cx0, cy0 = .62, 2.55, .66
    X_ = (U_ - V_ / 2) / (2 * np.pi) * sc + cx0
    Y_ = np.sqrt(3) / 2 * V_ / (2 * np.pi) * sc + cy0
    ax.pcolormesh(X_, Y_, B_[:-1, :-1], cmap='magma', shading='flat', rasterized=True)
    ax.text(cx0 + .15, .3, 'attractor sheet, 36 x 36\nunits, hexagonal recurrence', fontsize=fs - .4,
            ha='center', va='top', color='0.25')
    ax.text(.75, .32, 'landmarks\n(the task frame)', fontsize=fs, ha='center', va='center', bbox=box)
    # the gate: a switch, drawn open
    ax.plot([1.45, 1.62], [.32, .32], color='k', lw=1.0)
    ax.plot([1.62, 1.82], [.32, .44], color='k', lw=1.0)
    ax.plot([1.86, 1.98], [.32, .32], color='k', lw=1.0)
    ax.plot([1.62], [.32], 'o', ms=2.2, color='k')
    ax.text(1.62, .22, 'attentional\ngate', fontsize=fs - .4, ha='center', va='top', color='0.25')
    ax.annotate('', (2.22, .48), (1.98, .32), arrowprops=dict(arr, connectionstyle='arc3,rad=-.3'))
    ax.text(1.72, .62, 'correction', fontsize=fs - .8, ha='center', va='center', color='0.35',
            style='italic')
    # the bump read out: position, and a stop where it says the reward zone is
    ax.annotate('', (3.75, .66), (3.05, .66), arrowprops=arr)
    ax.text(4.55, .66, 'position decoded\nfrom the bump', fontsize=fs, ha='center', va='center', bbox=box)
    ax.annotate('', (5.95, .66), (5.35, .66), arrowprops=arr)
    ax.text(6.85, .66, 'slows and stops where\nit decodes the reward\nzone (90-110 cm)', fontsize=fs,
            ha='center', va='center', bbox=box)
    # what is analysed
    ax.text(8.0, 1.12, 'analysed as the recordings', fontsize=fs, weight='bold', ha='left', va='top',
            color='0.25')
    ax.text(8.0, .95, '36 of the 1,296 units sampled\nspikes at a recorded grid cell\'s\nrate and '
            'variability\nnoisy path integration\nthe gate opened and shut in\nblocks of trials',
            fontsize=fs - .6, ha='left', va='top', color='0.3', linespacing=1.35)


# Figure 9 stops after D-F: G-L are the two-session version's alone
fig = plt.figure(figsize=(9.6, 11.8) if SIM else (9.6, 13.8))
if SIM:                                     # the model's schematic is a row of its own, A
    gs0 = fig.add_gridspec(4, 1, hspace=.36, left=.045, right=.99, top=.955, bottom=.045,
                           height_ratios=[.72, 2.0, 1.95, 1.0])
    axSch = fig.add_subplot(gs0[0]); schematic(axSch)
    lp(axSch, 'A', x=-.01, y=1.04)
    axSch.text(0, 1.04, 'The gated attractor model (Figure 10), simulated', transform=axSch.transAxes,
               fontsize=8.4, weight='bold', va='bottom')
    gs = [gs0[1], gs0[2], gs0[3]]
else:
    gs = fig.add_gridspec(5, 1, hspace=.42, left=.045, right=.99, top=.945, bottom=.035,
                          height_ratios=[2.0, 1.95, 1.0, .95, 1.0])
LET = (lambda c: chr(ord(c) + 1)) if SIM else (lambda c: c)   # A is the schematic in Figure 9
EX_PV = []

for e_i, (mo, dy) in enumerate(EXAMPLES):
    is_sim = (mo, dy) == (0, 0)
    z = np.load(f'{PS}/attractor_sim/sim.npz' if is_sim else f'{PS}/attractor_cache/M{mo}D{dy}.npz')
    cells = list(z['cells'])
    mod = [int(c) for c in z['module']]
    mi = [cells.index(c) for c in mod]
    nb = int(z['nbin'])
    R = z['R'][mi].astype(float)
    P = z['P'][mi].astype(float)
    st = np.asarray(z['state']).astype(bool)
    ok, anch, non = (z[k].astype(bool) for k in ('ok', 'anch', 'non'))
    psi = np.stack([np.angle(np.exp(1j * q) @ P) for q in z['phi_all']]).reshape(3, -1, nb)
    adv = [abs(np.median([np.unwrap(r)[-1] - np.unwrap(r)[0] for r in w])) for w in psi]
    KA, KC = int(np.argmax(adv)), int(np.argmin(adv))
    if is_sim:                              # the simulated module's own labels
        # PC1 loadings exist only for units whose labels are not all missing
        _ld = np.full(len(z['labels']), np.nan)
        _ld[~np.isnan(z['labels']).all(1)] = z['pc1_load']
        Z = dict(labels=z['labels'], pc1=z['pc1'], pc1_load=_ld, cluster_id=z['cells'])
        sel = np.ones(len(z['cells']), bool)
    else:
        Z = np.load(f'{PS}/labels/M{mo}D{dy}.npz', allow_pickle=True)
        C = pd.read_csv(f'{PS}/anchoring_cells.csv')
        C = C[(C.mouse == mo) & (C.day == dy)]
        sel = np.isin(Z['cluster_id'], C.cluster_id[C.in_population].values)
    L, load = Z['labels'][sel], Z['pc1_load'][sel]
    with np.errstate(invalid='ignore'):
        varies = np.array([np.isfinite(r).sum() > 1 and np.nanstd(r) > 0 for r in L])
    L = L[np.where(varies)[0][np.argsort(load[varies])[::-1]]]
    pc1 = np.asarray(Z['pc1'])
    ntr = L.shape[1]

    blk = gs[e_i].subgridspec(2, 1, hspace=.42, height_ratios=[1.3, 1.3])
    # from left: where the mouse stopped, the state raster and PC1, the phase
    # maps, and the open-field maps with the cells on the sheet below them
    top = blk[0].subgridspec(1, 7, wspace=.18, width_ratios=[.5, .8, .2, .5, .5, .06, 2.33])   # psi maps as wide as the stops
    tt = (pd.DataFrame(dict(row=np.arange(len(st)), trial=np.arange(1, len(st) + 1), type='sim'))
          if is_sim else TT[(TT.mouse == mo) & (TT.day == dy)].sort_values('row'))
    unc, num = (tt.type == 'nb').values, tt.trial.values
    ax = fig.add_subplot(top[1])
    axRaster = ax
    ax.imshow(L.T, aspect='auto', cmap=TA, norm=NORM, interpolation='nearest',
              extent=[0, L.shape[0], ntr, 0])
    ax.set_xticks([]); ax.tick_params(labelleft=False, length=0)
    ax.set_xlabel(f'{L.shape[0]} ' + ('simulated grid cells' if is_sim else 'MEC cells'), fontsize=7.0)
    # where the mouse stopped on each trial (speed below 3 cm/s, the task's own
    # stop criterion, one dot per stop); in the recordings uncued trials only,
    # since on cued trials the zone is marked on the track
    axb = fig.add_subplot(top[0], sharey=axRaster)
    axb.axvspan(*RZ, color='0.88', lw=0)
    idx = z['idx']
    still = (z['spd'] < STOP_SPD) & (idx >= 0)
    on_ = np.where(still & ~np.r_[False, still[:-1]])[0]
    show = np.ones(len(st), bool) if is_sim else np.zeros(len(st), bool)
    if not is_sim:
        show[tt.row.values[unc]] = True
    on_ = on_[show[idx[on_]]]
    sp_, tr_ = z['trackpos'][on_], idx[on_]
    axb.scatter(sp_, tr_ + .5, s=2.2, c=[ANCH if st[t_] else NON for t_ in tr_], lw=0)
    axb.set_xlim(0, TL); axb.set_xticks([0, 100, 200])
    axb.tick_params(labelsize=6.0)
    axb.set_ylabel('trial', fontsize=7.2)
    axb.set_xlabel('stops (cm)', fontsize=6.8)
    hit = np.array([((sp_[tr_ == t_] >= RZ[0]) & (sp_[tr_ == t_] <= RZ[1])).any()
                    for t_ in range(len(st))])
    axb.set_title(('in the zone:' if is_sim else 'uncued, in the zone:'), fontsize=5.8, loc='left',
                  color='0.25', pad=15)
    for k_, (s_, c_, nm_) in enumerate(((True, ANCH, 'anch.'), (False, NON, 'non-anch.'))):
        m_ = show & (st == s_)
        axb.text(0, 1.012 + .05 * (1 - k_), f'{100 * hit[m_].mean():.0f}% {nm_}',
                 transform=axb.transAxes, fontsize=5.6, color=c_, va='bottom')
    print(f'    stops in the zone: {100 * hit[show & st].mean():.0f}% anchored, '
          f'{100 * hit[show & ~st].mean():.0f}% non-anchored ({show.sum()} trials)'
          + (f'; by gate {100 * hit[z["gate"].astype(bool)].mean():.0f}% open, '
             f'{100 * hit[~z["gate"].astype(bool)].mean():.0f}% shut' if is_sim else ''), flush=True)
    hy = 1.42 if e_i == 0 else 1.3          # B carries the torus insets above its maps
    lp(axb, LET('AB'[e_i]), x=-.62, y=hy)
    axb.text(0, hy, (f'Simulated: {len(mod)} grid cells of one gated attractor module, '
                     if is_sim else f'M{mo} D{dy}: {len(mod)} grid cells in the module, ') +
             f'{100 * st.mean():.0f}% of {len(st)} trials anchored', transform=axb.transAxes, fontsize=8.4,
             weight='bold', va='bottom')
    axp = fig.add_subplot(top[2], sharey=ax)
    tr = np.arange(len(pc1))
    axp.fill_betweenx(tr, 0, pc1, where=pc1 >= 0, color=ANCH, lw=0, edgecolor='none', interpolate=True)
    axp.fill_betweenx(tr, 0, pc1, where=pc1 < 0, color=NON, lw=0, edgecolor='none', interpolate=True)
    axp.axvline(0, color='0.4', lw=.5)
    axp.set_xticks([]); axp.tick_params(labelleft=False, length=0)
    for sp in axp.spines.values():
        sp.set_visible(False)
    axp.set_xlabel('PC1', fontsize=6.8)
    if is_sim:                              # the gate that drives it, trial by trial
        bt = matplotlib.transforms.blended_transform_factory(axp.transAxes, axp.transData)
        for t_, g_ in enumerate(z['gate']):
            if g_:
                axp.add_patch(plt.Rectangle((1.08, t_), .22, 1, transform=bt, color='k', lw=0,
                                            clip_on=False))
        axp.text(1.19, -.06, 'gate', transform=axp.transAxes, fontsize=6.0, ha='center',
                 va='top', color='0.25')
    for j, (k, lab) in enumerate(((KA, r'$\psi_\parallel$  along'), (KC, r'$\psi_\perp$  across'))):
        ax = fig.add_subplot(top[3 + j])
        ax.imshow(psi[k][:ntr], cmap='hsv', aspect='auto', interpolation='nearest',
                  extent=[0, TL, ntr, 0], vmin=-np.pi, vmax=np.pi)
        ax.set_yticks([]); ax.set_xticks([0, 100, 200]); ax.tick_params(labelsize=6.2)
        ax.set_xlabel('position (cm)', fontsize=7.0)
        ax.set_title(lab, fontsize=7.6, loc='left', color='0.25', pad=3)
        if e_i == 0:                        # which way round the torus each phase runs
            torus3d(fig, ax, j + 1)
    cax = fig.add_subplot(top[5])
    _cp = cax.get_position()                # half the maps' height, centred on them
    cax.set_position([_cp.x0, _cp.y0 + _cp.height / 4, _cp.width, _cp.height / 2])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-np.pi, np.pi), cmap='hsv'), cax=cax)
    cb.set_ticks([-np.pi, 0, np.pi]); cb.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
    cb.ax.tick_params(labelsize=5.6, length=2, pad=1.2); cb.outline.set_linewidth(.4)
    # the module in the open field, ordered by phase along the track
    maps = (z['of_maps'][[list(z['cells']).index(c) for c in mod]] if is_sim
            else of_maps(mo, dy, mod))
    order = np.argsort(z['phi_all'][KA])
    col_ = top[6].subgridspec(2, 1, hspace=.3, height_ratios=[1, 1])
    bx = fig.add_subplot(col_[0]); bb = bx.get_position(); bx.remove()
    W, H = fig.get_size_inches()
    best = max(range(6, 25), key=lambda c: min(bb.width * W / c, bb.height * H / np.ceil(len(mod) / c)))
    nr = int(np.ceil(len(mod) / best))
    side = min(bb.width * W / best, bb.height * H / nr) * .94
    for i, j in enumerate(order):
        a = fig.add_axes([bb.x0 + (i % best) * side / .94 / W,
                          bb.y1 - (i // best + 1) * side / .94 / H, side / W, side / H])
        a.imshow(maps[j], cmap='viridis', origin='lower', interpolation='nearest')
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        if i == 0:
            a.text(0, 1.08, 'the module in the open field (OF1), by phase along the track',
                   transform=a.transAxes, fontsize=7.0, color='0.25', va='bottom')
    # how the module's cells sample the sheet: each cell at its phases on the
    # along/across axes, carried onto the rhombic unit cell as the frames are
    # (Gardner et al. 2022), coloured by phase along as in B
    axR = fig.add_subplot(col_[1]); axR.set_anchor('W')
    u_, v_ = z['phi_all'][KA], z['phi_all'][KC]
    cu = np.array([-np.pi, np.pi, np.pi, -np.pi, -np.pi]); cv = np.array([-np.pi, -np.pi, np.pi, np.pi, -np.pi])
    axR.plot((cu - cv / 2) / (2 * np.pi), np.sqrt(3) / 2 * cv / (2 * np.pi), color='0.5', lw=.6)
    axR.scatter((u_ - v_ / 2) / (2 * np.pi), np.sqrt(3) / 2 * v_ / (2 * np.pi), c=u_, cmap='hsv',
                vmin=-np.pi, vmax=np.pi, s=9, lw=.3, edgecolor='k')
    axR.set_aspect('equal'); axR.axis('off')
    axR.set_title(f'the {len(mod)} cells on the sheet', fontsize=6.6, color='0.25', pad=2)
    # three consecutive uncued laps in each state, on the sheet: one row per lap,
    # one frame every 10 cm (the position-binned rates of that lap over those
    # 10 cm). The run is the one with the largest bump above shuffle over its
    # running bins, by the same rule in both states. Frames are spaced just
    # over one lattice vector apart, so neighbouring unit cells nearly tile.
    # Below, the animal's speed and the bump's (as Figure S14J: the two axes
    # that advance along the track, converted to cm/s by their phase advance
    # per cm over anchored trials), z-scored over the session's running bins
    # and averaged in 5 cm bins of each lap, one line per lap.
    coh = coherence(R, z['phi'])
    sh = np.mean([coherence(R, np.stack([rng.permutation(q) for q in z['phi']]))
                  for _ in range(NPERM)], 0)
    EX_PV.append((coh, sh, anch, non, mo, dy))
    Kn = kernel(z['phi_all'][[KA, KC]])
    Pt = P.reshape(len(mi), -1, nb)
    kk = list(np.argsort(adv)[::-1][:2])
    dpdx = np.array([np.median(np.diff(np.unwrap(psi[k][st], axis=1), axis=1)) / (TL / nb)
                     for k in kk])
    pt = np.stack([np.angle(np.exp(1j * q) @ R) for q in z['phi_all'][kk]])
    dd = (np.diff(pt, axis=1) + np.pi) % (2 * np.pi) - np.pi
    bs = np.r_[np.nan, (dd / dpdx[:, None]).mean(0) / .1]
    val = ok & np.r_[False, ok[:-1]] & np.isfinite(bs)
    # both speeds averaged over 2.5 s (attractor_speed_cycle.py): at 100 ms the
    # bump's speed is distorted by how unevenly the cells cover the torus, and
    # the distortion cancels over a lattice cycle (2.5 s in the simulation)
    bs = SC.boxcar(bs, val, SC.WIN)
    spW = SC.boxcar(z['spd'].astype(float), val, SC.WIN)
    val = val & np.isfinite(bs)
    zs = lambda v: (v - v[val].mean()) / (v[val].std() + 1e-9)
    ZA, ZB = zs(spW), zs(bs)
    tp = z['trackpos']
    bc = np.arange(2.5, TL, 5.0)
    low = blk[1].subgridspec(1, 4, wspace=.05, width_ratios=[1, 1, .13, .45])
    strips = [low[0], low[1]]
    chosen = {}
    for c_, (nm, col, sv) in enumerate((('anchored', ANCH, True), ('non-anchored', NON, False))):
        lo, hi = WIN.get((mo, dy, sv), (0, 10 ** 6))
        inwin = lambda i: lo <= i and i + 2 < hi
        runs = [i for i in range(len(tt) - 2) if unc[i:i + 3].all() and (st[i:i + 3] == sv).all()
                and num[i + 2] - num[i] == 2 and inwin(i)]
        kind = 'uncued'
        if not runs:                        # fall back to any three consecutive laps
            runs = [i for i in range(len(tt) - 2) if (st[i:i + 3] == sv).all()
                    and num[i + 2] - num[i] == 2 and inwin(i)]
            kind = 'consecutive'
        if (mo, dy, sv) in FIX:
            runs = [FIX[(mo, dy, sv)]]
            kind = ('uncued' if unc[runs[0]:runs[0] + 3].all() else
                    'consecutive (' + ', '.join('uncued' if u else 'cued'
                                                for u in unc[runs[0]:runs[0] + 3]) + ')')
        score = [np.mean((coh - sh)[ok & np.isin(idx, [i, i + 1, i + 2])]) for i in runs]
        r0 = runs[int(np.argmax(score))]
        chosen[sv] = r0
        # where these laps sit in the session: a triangle beside the raster
        axRaster.plot([-.035], [r0 + 1.5], marker='>', ms=5.5, color=col, mec='k', mew=.3,
                      transform=matplotlib.transforms.blended_transform_factory(
                          axRaster.transAxes, axRaster.transData), clip_on=False, zorder=6)
        ax = fig.add_subplot(strips[c_])
        ax.set_anchor('W')
        cx = lambda cm: (cm - POS[0]) / 10.0 * SP
        b = np.arange(nb) * TL / nb + TL / nb / 2
        for row in range(3):
            for k, cm in enumerate(POS):
                w = np.abs(b - cm) < 5
                Fm = (Kn * Pt[:, r0 + row][:, w].mean(1)[:, None, None]).sum(0)
                Fm = (Fm - Fm.min()) / (Fm.max() - Fm.min() + 1e-9)
                ax.pcolormesh(_X0 + cx(cm), _Y0 - row * RS, Fm, cmap='magma', shading='flat',
                              rasterized=True)
            ax.text(cx(0) - .85, -row * RS, f'trial {num[r0 + row]}', fontsize=5.4, ha='right',
                    va='center', color='0.3')
        # speeds, on the same horizontal scale as the frames
        yc, sc_ = -2 * RS - 2.0, .34
        for v_ in (-2, 0, 2):
            ax.plot([cx(0) - .6, cx(TL) - .4], [yc + v_ * sc_] * 2, color='0.85' if v_ else '0.6',
                    lw=.4, zorder=0)
            ax.text(cx(0) - .7, yc + v_ * sc_, f'{v_:+d}' if v_ else '0', fontsize=4.8, ha='right',
                    va='center', color='0.4')
        ax.text(cx(0) - 1.75, yc, 'speed\n(z)', fontsize=5.4, ha='center', va='center', color='0.3')
        for row in range(3):
            m = val & (idx == r0 + row)
            for V, cc, lw_ in ((ZA, '0.6', .7), (ZB, col, .9)):
                y = np.array([np.nanmean(V[m & (np.abs(tp - c) < 2.5)])
                              if (m & (np.abs(tp - c) < 2.5)).any() else np.nan for c in bc])
                ax.plot(cx(bc), yc + np.clip(y, -3.2, 3.2) * sc_, color=cc, lw=lw_, alpha=.9)
        for cm in (0, 50, 100, 150, 200):
            ax.text(cx(cm) - .5, yc - 3.4 * sc_, f'{cm}', fontsize=5.2, ha='center', va='top',
                    color='0.35')
        ax.text(cx(TL / 2) - .5, yc - 3.4 * sc_ - .38, 'position (cm)', fontsize=5.6, ha='center',
                va='top', color='0.35')
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_xlim(cx(0) - 2.15, cx(POS[-1]) + .8); ax.set_ylim(yc - 3.4 * sc_ - .8, .5)
        ax.set_title(f'{nm}: three {kind} trials (bump {max(score):+.3f})', fontsize=6.4,
                     loc='left', color=col, pad=2)
        ax.text(cx(TL) - .4, yc + 3.3 * sc_, 'grey, the animal; coloured, the bump',
                fontsize=5.0, ha='right', va='bottom', color='0.45')
    # what the bump says about position and speed on those laps. Position is
    # decoded from the bump's phase on all three axes by template matching: the
    # module's circular-mean phase in each 5 cm bin over anchored running bins,
    # the laps drawn here left out, and at each 100 ms bin the position whose
    # template is closest (largest summed cosine). Speed is the 5 cm averages
    # drawn under the frames.
    p3 = np.stack([np.angle(np.exp(1j * q) @ R) for q in z['phi_all']])
    shown = np.isin(idx, [chosen[s_] + i for s_ in chosen for i in range(3)])
    pbin = np.clip((tp / 5).astype(int), 0, len(bc) - 1)
    tmpl = np.stack([np.angle(np.exp(1j * p3[:, ok & anch & ~shown & (pbin == b_)]).mean(1))
                     for b_ in range(len(bc))], 1)                  # 3 x bins
    sc2 = low[3].subgridspec(2, 1, hspace=1.0)
    axP, axS = fig.add_subplot(sc2[0]), fig.add_subplot(sc2[1])
    tP, tS = [], []
    for sv, col in ((True, ANCH), (False, NON)):
        m = ok & np.isin(idx, [chosen[sv] + i for i in range(3)])
        dec = bc[np.argmax(np.cos(p3[:, m][:, :, None] - tmpl[:, None, :]).sum(0), 1)]
        axP.scatter(tp[m], dec, s=1.6, color=col, lw=0, alpha=.7, rasterized=True)
        tP.append(100 * np.mean(np.abs(dec - tp[m]) <= 20))
        xa, xb = [], []
        for row in range(3):
            mm = val & (idx == chosen[sv] + row)
            for c in bc:
                w_ = mm & (np.abs(tp - c) < 2.5)
                if w_.any():
                    xa.append(ZA[w_].mean()); xb.append(ZB[w_].mean())
        xa, xb = np.array(xa), np.array(xb)
        axS.scatter(xa, np.clip(xb, -4, 4), s=3, color=col, lw=0, alpha=.8)
        tS.append(np.corrcoef(xa, xb)[0, 1])
    axP.plot([0, TL], [0, TL], color='0.6', lw=.5, ls=(0, (3, 2)), zorder=0)
    axP.set_xlim(0, TL); axP.set_ylim(0, TL)
    axP.set_xticks([0, 100, 200]); axP.set_yticks([0, 100, 200])
    axP.set_xlabel('position (cm)', fontsize=6.0, labelpad=1)
    axP.set_ylabel('decoded from\nthe bump (cm)', fontsize=6.0, labelpad=1)
    axP.set_title('position from the bump; within 20 cm:', fontsize=5.8, loc='left', color='0.25',
                  pad=9)
    axP.text(0, 1.03, f'{tP[0]:.0f}% anchored', transform=axP.transAxes, fontsize=5.8, color=ANCH)
    axP.text(.55, 1.03, f'{tP[1]:.0f}% non-anch.', transform=axP.transAxes, fontsize=5.8, color=NON)
    axS.axhline(0, color='0.85', lw=.4, zorder=0); axS.axvline(0, color='0.85', lw=.4, zorder=0)
    axS.set_xlim(-3, 3); axS.set_ylim(-4, 4)
    axS.set_xlabel('animal speed (z)', fontsize=6.0, labelpad=1)
    axS.set_ylabel('bump speed (z)', fontsize=6.0, labelpad=1)
    axS.set_title('speed over 2.5 s (5 cm bins); r:', fontsize=5.8, loc='left', color='0.25', pad=9)
    axS.text(0, 1.03, f'{tS[0]:+.2f} anchored', transform=axS.transAxes, fontsize=5.8, color=ANCH)
    axS.text(.55, 1.03, f'{tS[1]:+.2f} non-anch.', transform=axS.transAxes, fontsize=5.8, color=NON)
    for a in (axP, axS):
        bare(a); a.tick_params(labelsize=5.4, length=2, pad=1)
    print(f'    decoded within 20 cm {tP[0]:.0f}% / {tP[1]:.0f}%; speed r {tS[0]:+.2f} / {tS[1]:+.2f}; '
          f'trials from rows {chosen[True]}, {chosen[False]}', flush=True)
    print(f'  M{mo} D{dy}: {len(mod)} grid cells, along/across axes {KA}/{KC}', flush=True)

# ---- D-F: the bump in every session (from the old Figure S14 G-J) ----------
D = pd.read_csv(f'{PS}/attractor_coherence.csv')
BT = pd.read_csv(f'{PS}/attractor_bump_by_trial.csv')
SPD = pd.read_csv(f'{PS}/attractor_speed_cycle.csv').drop(columns=['n_grid'])
R3 = gs[2].subgridspec(1, 5, wspace=.62, width_ratios=[1.6, 1, 1, 1, .42])

# C: how concentrated the population is, moment by moment, in each example
gC = R3[0].subgridspec(1, len(EX_PV) if SIM else 1, wspace=.35)
for j, (pv, shv, a, n_, mo_, dy_) in enumerate(EX_PV[:2] if SIM else EX_PV[:1]):
    ax = fig.add_subplot(gC[j])
    if not j:
        lp(ax, LET('C'), x=-.45 if SIM else -.3)
    bins = np.linspace(min(pv.min(), shv.min()), max(pv.max(), shv.max()), 40)
    for v, c, lab in ((shv[a | n_], '0.55', 'shuffle'), (pv[n_], NON, 'non-anch.'),
                      (pv[a], ANCH, 'anchored')):
        ax.hist(v, bins=bins, density=True, histtype='step', color=c, lw=1.0, label=lab)
    print(f'  {"simulated" if (mo_, dy_) == (0, 0) else f"M{mo_} D{dy_}"}: bump above shuffle '
          f'{pv[a].mean() - shv[a].mean():+.3f} anchored, {pv[n_].mean() - shv[n_].mean():+.3f} '
          f'non-anchored', flush=True)
    bare(ax); ax.tick_params(labelsize=5.8)
    ax.set_xlabel('bump population\nvector, per 100 ms', fontsize=6.2)
    if not j:
        ax.set_ylabel('density', fontsize=6.8)
    if j == len(EX_PV[:2]) - 1 or not SIM:
        ax.legend(fontsize=5.0, frameon=False, loc='upper right', handlelength=.9, borderpad=.1,
                  labelspacing=.2)
    ax.set_title('simulated' if (mo_, dy_) == (0, 0) else f'M{mo_} D{dy_}', fontsize=6.6,
                 loc='left', color='0.25')

# D: every session, against its own shuffle
ax = fig.add_subplot(R3[1]); lp(ax, LET('D'), x=-.3)
for _, r in D.iterrows():
    ax.plot([0, 1], [r.bump_anch, r.bump_non], color='0.78', lw=.6, zorder=1)
_big = D.n_grid.values >= MIN_RELIABLE
for x, col, c in ((0, 'bump_anch', ANCH), (1, 'bump_non', NON)):
    v = D[col].values
    jx = np.full(len(v), x) + rng.normal(0, .03, len(v))
    ax.scatter(jx[_big], v[_big], s=9, color=c, lw=0, zorder=2)
    ax.scatter(jx[~_big], v[~_big], s=9, facecolor='none', edgecolor=c, lw=.6, zorder=2)
    ax.plot([x - .18, x + .18], [v.mean()] * 2, color=c, lw=1.8, zorder=3)
ax.axhline(0, color='k', lw=.8, ls=(0, (3, 2)))
ax.set_xlim(-.45, 1.45); ax.set_xticks([0, 1])
ax.set_xticklabels(['anchored', 'non-\nanchored'], fontsize=6.4)
ax.set_ylabel('bump above shuffle', fontsize=6.8)
bare(ax)
pa, pn_ = wilcoxon(D.bump_anch).pvalue, wilcoxon(D.bump_non).pvalue
pd_ = wilcoxon(D.bump_anch - D.bump_non).pvalue
psp = wilcoxon(D.sm_anch - D.sm_non).pvalue
ax.set_ylim(-.004, D.bump_anch.max() * 1.45)
ax.text(.5, .99, f'both above zero: {star(max(pa, pn_))}\nanchored higher: {star(pd_)}\n'
        f'speeds equated: {star(psp)}', transform=ax.transAxes, fontsize=5.8, ha='center',
        va='top', color='0.25')
ax.set_title(f'{len(D)} sessions, {int(D.n_grid.sum())} grid cells', fontsize=6.8, loc='left',
             color='0.25')

# the bump across the switch, trial by trial (not shown; Methods)
WLAG = 8


def _epochs(a):
    """State epochs, with runs shorter than three trials absorbed into the
    preceding one -- the definition of a transition used everywhere else."""
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


segs, slopes = [], []
for (mo_s, dy_s), g in BT.groupby(['mouse', 'day']):
    g = g.sort_values('trial')
    tr_, bb_, aa_ = g.trial.values, g.bump.values, g.anch.values
    full = np.full(tr_.max() + 1, np.nan); full[tr_] = bb_
    fa = np.zeros(tr_.max() + 1, bool); fa[tr_] = aa_
    ep = _epochs(fa)
    for i, j, state in ep:
        if j - i >= 8:
            seg = full[i:j]; kk_ = np.arange(j - i); o = np.isfinite(seg)
            if o.sum() >= 6:
                slopes.append((state, np.polyfit(kk_[o], seg[o], 1)[0]))
    for i, j, state in ep[1:]:
        if state or not fa[i - 1]:              # anchored -> non-anchored only
            continue
        if i - WLAG < 0 or i + WLAG > len(full):
            continue
        q = full[i - WLAG:i + WLAG]
        if np.isfinite(q).sum() >= 1.4 * WLAG:
            segs.append(q)
SEG = np.array(segs)
step = np.nanmean(SEG[:, WLAG:WLAG + 3], 1) - np.nanmean(SEG[:, WLAG - 3:WLAG], 1)
p_step = wilcoxon(step[np.isfinite(step)]).pvalue
sl_non = np.array([v for s_, v in slopes if not s_])
p_slope = wilcoxon(sl_non).pvalue
print(f'  across the switch (not shown): {len(SEG)} switches, step {np.nanmean(step):+.4f} '
      f'(p = {p_step:.2g}), slope within non-anchored epochs {np.median(sl_non):+.5f} '
      f'(p = {p_slope:.2g})', flush=True)
# the rest of the trial-by-trial account in Methods: the per-trial bump by state,
# the drift within anchored epochs, and the state difference by each trial's
# distance in trials from the nearest switch (sessions with a switch)
_pt = BT.groupby(['mouse', 'day', 'anch']).bump.mean().unstack()
sl_anc = np.array([v for s_, v in slopes if s_])
print(f'  per-trial bump {_pt[True].mean():+.4f} anchored, {_pt[False].mean():+.4f} non-anchored '
      f'(p = {wilcoxon(_pt[True] - _pt[False]).pvalue:.2g}, {len(_pt)} sessions); slope within '
      f'anchored epochs {np.median(sl_anc):+.5f} (p = {wilcoxon(sl_anc).pvalue:.2g})', flush=True)
_rows = []
for (mo_s, dy_s), g in BT.groupby(['mouse', 'day']):
    g = g.sort_values('trial')
    fa = np.zeros(g.trial.max() + 1, bool); fa[g.trial.values] = g.anch.values
    ep = _epochs(fa)
    if len(ep) < 2:
        continue
    st = np.zeros(len(fa), bool)
    for i, j, s_ in ep:
        st[i:j] = s_
    sw = np.array([i for i, _, _ in ep[1:]])
    for t, b in zip(g.trial.values, g.bump.values):
        _rows.append((mo_s, dy_s, st[t], int(np.min(np.where(t >= sw, t - sw, sw - t - 1))) + 1, b))
_R = pd.DataFrame(_rows, columns=['mouse', 'day', 'anch', 'dist', 'bump'])
for lo, hi in ((1, 2), (3, 5), (6, 10), (11, 10**6)):
    m_ = (_R[_R.dist.between(lo, hi)].groupby(['mouse', 'day', 'anch']).bump.mean()
          .unstack().dropna())
    d_ = m_[True] - m_[False]
    print(f'  {lo}-{hi if hi < 10**6 else ""} trials from a switch: anchored - non-anchored '
          f'{d_.mean():+.4f} (p = {wilcoxon(d_).pvalue:.2g}, {len(d_)} sessions)', flush=True)

# E: the bump's speed against the animal's, every session
SPD = SPD.merge(D[['mouse', 'day', 'n_grid']], on=['mouse', 'day'])
ax = fig.add_subplot(R3[2]); lp(ax, LET('E'), x=-.3)
for _, r in SPD.iterrows():
    ax.plot([0, 1], [r.r_a, r.r_n], color='0.78', lw=.6, zorder=1)
_bg = SPD.n_grid.values >= MIN_RELIABLE
for x_, v, c in ((0, SPD.r_a.values, ANCH), (1, SPD.r_n.values, NON)):
    jx = np.full(len(v), x_) + rng.uniform(-.06, .06, len(v))
    ax.scatter(jx[_bg], v[_bg], s=9, color=c, lw=0, zorder=2)
    ax.scatter(jx[~_bg], v[~_bg], s=9, facecolor='none', edgecolor=c, lw=.6, zorder=2)
    ax.plot([x_ - .2, x_ + .2], [np.median(v)] * 2, color=c, lw=2, zorder=3)
_nul = np.r_[SPD.rnull_a, SPD.rnull_n].mean()
ax.axhline(_nul, color='0.3', lw=.8, ls=(0, (3, 2)))
ax.text(1.42, _nul, 'shuffle', fontsize=5.4, color='0.3', va='center', ha='left')
p_ea = wilcoxon(SPD.r_a - SPD.rnull_a).pvalue
p_en = wilcoxon(SPD.r_n - SPD.rnull_n).pvalue
p_d = wilcoxon(SPD.r_a, SPD.r_n).pvalue
p_m = wilcoxon(SPD.rm_a, SPD.rm_n).pvalue
ax.set_xticks([0, 1]); ax.set_xticklabels(['anchored', 'non-\nanchored'], fontsize=6.4)
ax.set_xlim(-.45, 1.45)
ax.set_ylabel('r, bump speed with the\nanimal\'s, over 2.5 s', fontsize=6.8)
ax.text(.5, .99, f'both above shuffle: {star(max(p_ea, p_en))}\nanchored higher: {star(p_d)}\n'
        f'speeds matched: {star(p_m)}', transform=ax.transAxes, fontsize=5.8, ha='center',
        va='top', color='0.25')
ax.set_ylim(min(-.01, SPD[['r_a', 'r_n']].values.min() - .02), SPD.r_a.max() * 1.45)
bare(ax)
ax.set_title(f'{len(SPD)} sessions: the bump keeps\npace, less closely when lost', fontsize=6.8,
             loc='left', color='0.25')

# F: the bump's speed gain, every session: the slope of its speed on the
# animal's (bump cm per cm run), 1 being perfect integration. Over 100 ms bins
# (attractor_coherence.bump_speed): the readout's distortion is noise in the
# bump's speed, which lowers r (hence the 2.5 s windows of E) but does not bias
# a slope on the animal's speed; the 2.5 s slopes, from far fewer windows, are
# printed beside it
G100 = (pd.read_csv(f'{PS}/attractor_speed.csv')[['mouse', 'day', 'slope_a', 'slope_n']]
        .merge(D[['mouse', 'day', 'n_grid']], on=['mouse', 'day']))
ax = fig.add_subplot(R3[3]); lp(ax, LET('F'), x=-.3)
for _, r in G100.iterrows():
    ax.plot([0, 1], [r.slope_a, r.slope_n], color='0.78', lw=.6, zorder=1)
_bg1 = G100.n_grid.values >= MIN_RELIABLE
for x_, v, c in ((0, G100.slope_a.values, ANCH), (1, G100.slope_n.values, NON)):
    jx = np.full(len(v), x_) + rng.uniform(-.06, .06, len(v))
    ax.scatter(jx[_bg1], v[_bg1], s=9, color=c, lw=0, zorder=2)
    ax.scatter(jx[~_bg1], v[~_bg1], s=9, facecolor='none', edgecolor=c, lw=.6, zorder=2)
    ax.plot([x_ - .2, x_ + .2], [np.median(v)] * 2, color=c, lw=2, zorder=3)
ax.axhline(1, color='0.3', lw=.8, ls=(0, (3, 2)))
ax.text(1.42, 1, 'gain 1', fontsize=5.4, color='0.3', va='center', ha='left')
ax.axhline(0, color='0.75', lw=.5, zorder=0)
g_a1, g_n1 = wilcoxon(G100.slope_a - 1).pvalue, wilcoxon(G100.slope_n - 1).pvalue
g_d = wilcoxon(G100.slope_a, G100.slope_n).pvalue
g_mm, g_mp = mixed(G100.set_index(['mouse', 'day']).eval('slope_a - slope_n'))
ax.set_xticks([0, 1]); ax.set_xticklabels(['anchored', 'non-\nanchored'], fontsize=6.4)
ax.set_xlim(-.45, 1.45)
ax.set_ylabel('gain, bump cm per cm run', fontsize=6.8)
ax.set_ylim(min(-.3, G100[['slope_a', 'slope_n']].values.min() - .3),
            G100[['slope_a', 'slope_n']].values.max() * 1.45)
ax.text(.5, .99, f'anchored vs 1: {star(g_a1)}\nnon-anchored vs 1: {star(g_n1)}\n'
        f'anchored higher: {star(g_mp)}', transform=ax.transAxes,
        fontsize=5.8, ha='center', va='top', color='0.25')
bare(ax)
ax.set_title(f'the bump keeps full pace when\nanchored ({G100.slope_a.median():.2f}), '
             f'not when lost ({G100.slope_n.median():.2f})', fontsize=6.8, loc='left', color='0.25')
print(f'  gain over 100 ms: median {G100.slope_a.median():.3f} anchored, {G100.slope_n.median():.3f} '
      f'non-anchored; vs 1 p = {g_a1:.2g}, {g_n1:.2g}; anchored higher in '
      f'{int((G100.slope_a > G100.slope_n).sum())} of {len(G100)}, Wilcoxon p = {g_d:.2g}, mixed '
      f'{g_mm:+.3f}, p = {g_mp:.2g}', flush=True)
# the key to E-G: state by colour, module size by fill
from matplotlib.lines import Line2D                               # noqa: E402
axL = fig.add_subplot(R3[4]); axL.axis('off')
_h = [Line2D([], [], ls='', marker='o', ms=4.2, color=ANCH, mew=0, label='anchored'),
      Line2D([], [], ls='', marker='o', ms=4.2, color=NON, mew=0, label='non-anchored'),
      Line2D([], [], ls='', marker='o', ms=4.2, color='0.35', mew=0,
             label=f'session, at least\n{MIN_RELIABLE} grid cells'),
      Line2D([], [], ls='', marker='o', ms=4.2, mfc='none', mec='0.35', mew=.7,
             label=f'session, fewer\nthan {MIN_RELIABLE} grid cells'),
      Line2D([], [], color='0.78', lw=.8, label='same session'),
      Line2D([], [], color='0.4', lw=2, label=f'mean ({LET("D")}),\nmedian ({LET("E")}, {LET("F")})')]
axL.legend(handles=_h, loc='center left', bbox_to_anchor=(-.15, .5), fontsize=5.8, frameon=False,
           handlelength=1.3, handletextpad=.5, labelspacing=.9, borderaxespad=0)
_p25 = mixed(SPD.set_index(['mouse', 'day']).eval('slope_a - slope_n'))[1]
print(f'  gain over 2.5 s: median {SPD.slope_a.median():.3f} anchored, {SPD.slope_n.median():.3f} '
      f'non-anchored; vs 1 p = {wilcoxon(SPD.slope_a - 1).pvalue:.2g}, '
      f'{wilcoxon(SPD.slope_n - 1).pvalue:.2g}; Wilcoxon p = '
      f'{wilcoxon(SPD.slope_a, SPD.slope_n).pvalue:.2g}, mixed p = {_p25:.2g}', flush=True)
_sg = SC.session(np.load(f'{PS}/attractor_sim/sim.npz'), np.random.default_rng(0))
print(f'  simulated gain: over 100 ms {_sg["slope100_a"]:.2f} anchored, {_sg["slope100_n"]:.2f} '
      f'non-anchored; over 2.5 s {_sg["slope_a"]:.2f}, {_sg["slope_n"]:.2f}',
      flush=True)

if SIM:
    fig.savefig(OUT, dpi=250, bbox_inches='tight')
    print('wrote', OUT)
    sys.exit()

# ---- G, H: how many grid cells the measure needs ---------------------------
RL = pd.read_csv(f'{PS}/attractor_reliability.csv')
AG = RL.dropna(subset=['agree_a']).assign(ag=lambda d: (d.agree_a + d.agree_n) / 2)
R4 = gs[3].subgridspec(1, 2, wspace=.25)
ax = fig.add_subplot(R4[0]); lp(ax, 'G', x=-.13)
for (m__, d__), g in AG.groupby(['mouse', 'day']):
    big = g.n_grid.iloc[0] >= MIN_RELIABLE
    ax.plot(g.k, g.ag, color='0.2' if big else '0.72', lw=.9 if big else .8, zorder=2 if big else 1)
    ax.plot(g.k.iloc[-1], g.ag.iloc[-1], 'o', ms=2.6, color='0.2' if big else '0.72', zorder=3)
ax.axhline(0, color='k', lw=.6, ls=(0, (3, 2)))
ax.set_xlabel('grid cells in each half', fontsize=6.8)
ax.set_ylabel('split-half agreement\nof the bump\'s phase', fontsize=6.8)
ax.set_xlim(1.5, AG.k.max() + .5)
bare(ax)
k4 = AG[AG.k == 4].groupby(['mouse', 'day']).agg(ag=('ag', 'first'), n=('n_grid', 'first'))
rho4, prho4 = spearmanr(k4.n, k4.ag)
ax.set_title(f'agreement grows with the cells per half; at a matched 4 per half, sessions\n'
             f'with small modules are no worse (rho = {rho4:+.2f}, p = {prho4:.2f})',
             fontsize=6.8, loc='left', color='0.25')
ax.text(.99, .05, f'dark: modules of at least {MIN_RELIABLE} grid cells; dot, half the module',
        transform=ax.transAxes, fontsize=5.6, ha='right', color='0.35')
ax = fig.add_subplot(R4[1]); lp(ax, 'H', x=-.13)
SB = AG.groupby(['mouse', 'day']).apply(lambda g: g.ag.iloc[-1], include_groups=False)
SB = (2 * SB / (1 + SB)).rename('sb').reset_index().merge(D[['mouse', 'day', 'n_grid']])
ax.scatter(SB.n_grid, SB.sb, s=14, c=['0.2' if n >= MIN_RELIABLE else '0.72' for n in SB.n_grid],
           lw=0, zorder=3)
ax.axvline(MIN_RELIABLE - .5, color='k', lw=.7, ls=(0, (3, 2)))
ax.set_xlabel('grid cells in the module', fontsize=6.8)
ax.set_ylabel('reliability of the\nwhole module (est.)', fontsize=6.8)
ax.set_ylim(0, .7)
bare(ax)
Db = D[D.n_grid >= MIN_RELIABLE]
Sb = SPD[SPD.n_grid >= MIN_RELIABLE]
p_b = wilcoxon(Db.bump_anch - Db.bump_non).pvalue
p_r = smf.mixedlm('d ~ 1', Sb.assign(d=Sb.r_a - Sb.r_n), groups=Sb['mouse']).fit(
    reml=True).pvalues['Intercept']
ax.set_title(f'{len(Db)} sessions ({Db.mouse.nunique()} mice) have at least {MIN_RELIABLE} grid cells. '
             f'In them alone: both states above\nshuffle '
             f'{star(max(wilcoxon(Db.bump_anch).pvalue, wilcoxon(Db.bump_non).pvalue))}; anchored '
             f'higher {star(p_b)}; speed r higher {star(p_r)} (mouse random effect)',
             fontsize=6.8, loc='left', color='0.25')
print(f'  bump anchored {D.bump_anch.mean():+.4f} p={pa:.2g}; non {D.bump_non.mean():+.4f} '
      f'p={pn_:.2g}; difference p={pd_:.2g}; {len(SEG)} switches, step p={p_step:.3g}, '
      f'slope p={p_slope:.3g}; >= {MIN_RELIABLE}: {len(Db)} sessions, p_b {p_b:.2g}, p_r {p_r:.2g}')

# ---- I-L: one bump or two? -------------------------------------------------
MB = pd.read_csv(f'{PS}/attractor_multibump.csv')
A_ = MB[MB.state == 'anchored'].set_index(['mouse', 'day'])
N_ = MB[MB.state == 'non-anchored'].set_index(['mouse', 'day'])
Fz = np.load(f'{PS}/attractor_multibump_frames.npz')
F = pd.DataFrame({k: Fz[k] for k in Fz.files})
G = gs[4].subgridspec(1, 4, wspace=.62, width_ratios=[1.15, 1, 1, 1.1])

# D: second-peak height against bump strength (deciles within session)
ax = fig.add_subplot(G[0]); lp(ax, 'I', x=-.27)
cur = {True: [], False: []}; curn = {True: [], False: []}; xs = {True: [], False: []}
diff = []
for (mo, dy), g in F.groupby(['mouse', 'day']):
    q = pd.qcut(g.bump, 10, labels=False, duplicates='drop')
    if q.max() < 9:
        continue
    d_ = []
    for s_ in (True, False):
        h = g[g.state == s_]
        qq = q[g.state == s_]
        cur[s_].append([h.p2[qq == b].mean() for b in range(10)])
        curn[s_].append([h.p2_null[qq == b].mean() for b in range(10)])
        xs[s_].append([h.bump[qq == b].mean() for b in range(10)])
    for b in range(10):
        a, n = g[(g.state) & (q == b)], g[(~g.state) & (q == b)]
        if len(a) > 30 and len(n) > 30:
            d_.append(n.p2.mean() - a.p2.mean())
    diff.append(dict(mouse=mo, day=dy, matched=np.mean(d_)))
for s_, c in ((True, ANCH), (False, NON)):
    y = np.array(cur[s_]); x = np.nanmean(np.array(xs[s_]), 0)
    m, se = np.nanmean(y, 0), np.nanstd(y, 0) / np.sqrt(np.isfinite(y).sum(0))
    ax.fill_between(x, m - se, m + se, color=c, alpha=.25, lw=0, edgecolor='none')
    ax.plot(x, m, color=c, lw=1.2, label='anchored' if s_ else 'non-anchored')
    ax.plot(x, np.nanmean(np.array(curn[s_]), 0), color=c, lw=.8, ls=(0, (3, 2)))
ax.set_xlabel('bump strength above shuffle\n(within-session deciles)', fontsize=6.8)
ax.set_ylabel('second peak, fraction of\nthe top one', fontsize=6.8)
ax.legend(fontsize=5.8, frameon=False, loc='upper right')
r_ = [np.corrcoef(g.bump, g.p2)[0, 1] for _, g in F.groupby(['mouse', 'day'])]
ax.set_title(f'weaker bump, higher second peak, same\nin both states (dashed, phase shuffle);\n'
             f'r = {np.median(r_):.2f}, negative in {sum(np.array(r_) < 0)} of {len(r_)}',
             fontsize=6.4, loc='left', color='0.25')
bare(ax)

# E: the state difference, raw and at matched strength
ax = fig.add_subplot(G[1]); lp(ax, 'J', x=-.3)
M = pd.DataFrame(diff).set_index(['mouse', 'day'])
raw = (N_.p2 - A_.p2).reindex(M.index)
dB = (N_.bump - A_.bump).reindex(M.index)
for a_, b_ in zip(raw, M.matched):
    ax.plot([0, 1], [a_, b_], color='0.85', lw=.6)
for x, v in ((0, raw), (1, M.matched)):
    ax.scatter(np.full(len(v), x), v, s=8, color='k', lw=0, zorder=3)
    ax.plot([x - .2, x + .2], [np.median(v)] * 2, color='k', lw=2)
ax.axhline(0, color='k', lw=.6, ls=(0, (3, 2)))
ax.set_xticks([0, 1]); ax.set_xticklabels(['all frames', 'matched\nstrength'], fontsize=6.4)
ax.set_xlim(-.5, 1.5)
ax.set_ylabel('second peak,\nnon-anchored minus anchored', fontsize=6.8)
m1, p1 = mixed(raw); m2, p2 = mixed(M.matched); m3, p3 = mixed(dB)
ax.set_title(f'no state difference: {m1:+.3f}\n({star(p1)}); matched {m2:+.3f} ({star(p2)});\n'
             f'bump weaker by {-m3:.3f} ({star(p3)})', fontsize=6.4, loc='left',
             color='0.25')
bare(ax)

# F: strong second peaks against the shuffle
ax = fig.add_subplot(G[2]); lp(ax, 'K', x=-.3)
for s_i, (T, c) in enumerate(((A_, ANCH), (N_, NON))):
    x0 = s_i * 2.4
    for a_, b_ in zip(T.two, T.two_null):
        ax.plot([x0, x0 + 1], [100 * a_, 100 * b_], color='0.85', lw=.6)
    for x, v, al in ((x0, T.two, 1), (x0 + 1, T.two_null, .45)):
        ax.scatter(np.full(len(v), x), 100 * v, s=8, color=c, lw=0, alpha=al, zorder=3)
        ax.plot([x - .2, x + .2], [100 * np.median(v)] * 2, color=c, lw=2)
ax.set_xticks([0, 1, 2.4, 3.4])
ax.set_xticklabels(['data', 'shuffle', 'data', 'shuffle'], fontsize=6.4)
ax.set_ylabel('% frames with a second\npeak at least 0.8 of the top', fontsize=6.8)
ma, pa = mixed(A_.two - A_.two_null); mn, pn = mixed(N_.two - N_.two_null)
mt, pt = mixed(N_.two - A_.two)
ax.set_title(f'above the shuffle in both states\n(+{100 * ma:.0f}, +{100 * mn:.0f} points); not more\n'
             f'when non-anchored ({100 * mt:+.1f}, {star(pt)})', fontsize=6.4, loc='left',
             color='0.25')
bare(ax)

# G: is the second peak a leftover anchored bump? Distance (on the sheet) from
# where position puts the bump -- the module's mean phase at that 5 cm on
# anchored trials -- minus the distance of a uniformly placed point
ax = fig.add_subplot(G[3]); lp(ax, 'L', x=-.22)
cols = [('d_top', A_, 'top peak,\nanchored', ANCH, 1), ('d_top', N_, 'top peak,\nnon-anch.', NON, 1),
        ('d_second', N_, 'second peak,\nnon-anch.', NON, .45)]
tt = []
for x, (k, T, lab, c, al) in enumerate(cols):
    v = T[k] - T.d_chance
    ax.scatter(np.full(len(v), x) + rng.uniform(-.1, .1, len(v)), v, s=8, color=c, lw=0,
               alpha=al, zorder=3)
    ax.plot([x - .25, x + .25], [np.median(v)] * 2, color=c, lw=2)
    m_, p_ = mixed(v)
    tt.append(f'{m_:+.2f} ({star(p_)})')
ax.axhline(0, color='k', lw=.6, ls=(0, (3, 2)))
ax.set_xticks(range(3)); ax.set_xticklabels([c[2] for c in cols], fontsize=6.0)
ax.set_xlim(-.6, 2.6)
ax.set_ylabel('distance from where position\nputs the bump, minus chance (rad)', fontsize=6.6)
ax.set_title(f'the second peak is not a leftover\nanchored bump: {tt[0]}, {tt[1]},\n{tt[2]}',
             fontsize=6.4, loc='left', color='0.25')
bare(ax)

fig.savefig(OUT, dpi=250, bbox_inches='tight')
print('wrote', OUT)

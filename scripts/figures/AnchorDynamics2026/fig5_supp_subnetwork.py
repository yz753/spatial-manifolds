'''Figure S18: do grid cells and interneurons form a subnetwork that follows the
anchoring state? Related to Figure 5.

As identities they do: grid-interneuron pairs share the state more than other
pairs of the same session (Figure 2G), and they are wired together (Figure 5E).
The question here is whether the WIRING picks out the cells that follow, i.e.
whether the cells joined by detected monosynaptic connections are a distinct,
state-following subnetwork within those identities.

    A  one session's grid-interneuron network on the probe (NP2 4-shank, recorded
       contacts darker than inactive ones): grid cells and
       interneurons coloured by agreement with the MEC PC1, joined by their
       detected connections
    B  identity-level co-following: pair label correlation per session, by
       identity pair, against the session's other pairs
    C  pooled across sessions, wired cells (at least one detected connection to
       another grid cell or interneuron) look like stronger followers
    D  within session they are not: wired minus unwired agreement per session,
       and the session-fixed-effect model with log spike count
    E  number of grid/interneuron partners against agreement, both residualised
       on session and log spike count
    F  the wired cells' labels are no more coherent with each other than
       identity- and spike-count-matched sets drawn from all of the session's
       grid cells and interneurons
    G-I direction of the connections along the probe (all MEC connections, every
       identity): inhibitory connections run ventral to dorsal more often, and
       long-range excitatory ones dorsal to ventral, than the cells' positions
       predict

Direction null: a connection's presynaptic cell has candidate partners above
and below it, and near the probe ends most lie on one side, so the expected
fraction pointing dorsal is, per connection, the fraction of the session's
tested cells at the same distance (+-DIST_TOL um) from the presynaptic cell that
lie dorsal to it. Larger probe y is dorsal.

Detection depends on spike count and spike count predicts following (Figure
4I-K), so every within-session comparison controls for log spike count.
Cells: MEC (rule anatomy) grid cells and putative interneurons tested for
connections (data/monosyn/cells2_w*.csv), agreement = pc1_by_region excess
(switching cells only; locked cells have none).

Writes fig5_supp_subnetwork.pdf; caches data/monosyn/subnetwork_{cells,coherence}.csv
and data/monosyn/connection_direction.csv
'''
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import spearmanr, wilcoxon

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
sys.path.insert(0, f'{ROOT}/src')
from spatial_manifolds.anchoring import load_session_labels            # noqa: E402

plt.rcParams['font.family'] = 'Arial'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
MS, PS = f'{ROOT}/data/monosyn', f'{ROOT}/data/population_state'
OUT = f'{FIG}/fig5_supp_subnetwork.pdf'
GI = ('grid', 'putative interneuron')
ICOL = {'grid': '#c04744', 'putative interneuron': '#d95f02'}
EXC_C, INH_C = '#1b998b', '#6d2e9e'
AGREE_CMAP = LinearSegmentedColormap.from_list(
    'agree', ['#3a3a3a', '#9a9a9a', '#ffffff', '#f4a259', '#b35806'])
EXAMPLE = (29, 23)
N_NULL = 1000
DIST_TOL = 25
KIND_C = {'inh': '#6d2e9e', 'exc': '#1b998b'}
N_ANG = 16                     # polar bins, angle of post from pre in the probe plane


def tables():
    U = pd.read_csv(f'{PS}/unit_table.csv')
    ID = {(m, d, c): i for m, d, c, i in zip(U.mouse, U.day, U.cluster_id, U.identity)}
    CN = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{MS}/connections2_w*.csv'))])
    K = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{MS}/cells2_w*.csv'))])
    R = pd.read_csv(f'{PS}/pc1_by_region.csv')
    R = R[R.is_mec]
    return ID, CN, K, R


def cell_table(ID, CN, K, R):
    rows = []
    for (mo, dy), k in K.groupby(['mouse', 'day']):
        p = R[(R.mouse == mo) & (R.day == dy)].set_index('cluster_id')
        ids = [int(c) for c in k.cluster_id if int(c) in p.index and ID.get((mo, dy, int(c))) in GI]
        if not ids:
            continue
        cc = CN[(CN.mouse == mo) & (CN.day == dy)]
        cc = cc[cc.pre.isin(ids) & cc.post.isin(ids)]
        part = {c: set() for c in ids}
        for a, b in zip(cc.pre.astype(int), cc.post.astype(int)):
            part[a].add(b); part[b].add(a)
        kk = k.set_index('cluster_id')
        for c in ids:
            rows.append(dict(mouse=mo, day=dy, cell=c, ident=ID[(mo, dy, c)], deg=len(part[c]),
                             member=len(part[c]) > 0, n_spikes=kk.n_spikes.get(c),
                             px=kk.px.get(c), py=kk.py.get(c), excess=p.excess.get(c)))
    D = pd.DataFrame(rows)
    D['sess'] = D.mouse.astype(str) + '_' + D.day.astype(str)
    D['lns'] = np.log(D.n_spikes)
    return D


def direction_table(CN, K):
    """Each MEC-MEC connection's dorsal offset and its expected chance of pointing dorsal."""
    C = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
    reg = {(m, d, c): str(r) for m, d, c, r in zip(C.mouse, C.day, C.cluster_id, C.brain_region)}
    rows = []
    for (mo, dy), cc in CN.groupby(['mouse', 'day']):
        k = K[(K.mouse == mo) & (K.day == dy)].set_index('cluster_id')
        for a, b, kind in zip(cc.pre.astype(int), cc.post.astype(int), cc.kind):
            if not (reg.get((mo, dy, a), '').startswith('ENTm') and reg.get((mo, dy, b), '').startswith('ENTm')):
                continue
            x0, y0, x1, y1 = k.px[a], k.py[a], k.px[b], k.py[b]
            dist = np.hypot(x1 - x0, y1 - y0)
            if y1 == y0:
                continue
            r = np.hypot(k.px - x0, k.py - y0)
            cand = (np.abs(r - dist) < DIST_TOL) & (k.index != a) & (k.py != y0)
            # the same candidates' angles from the presynaptic cell, as a distribution over bins
            ang = np.arctan2(k.py[cand] - y0, k.px[cand] - x0) % (2 * np.pi)
            h = np.histogram(ang, N_ANG, (0, 2 * np.pi))[0] / max(cand.sum(), 1)
            rows.append(dict(mouse=mo, day=dy, pre=a, post=b, kind=kind, dx_um=x1 - x0, dy_um=y1 - y0,
                             dist=dist, up=float(y1 > y0), expected=(k.py[cand] > y0).mean(),
                             **{f'a{i}': v for i, v in enumerate(h)}))
    return pd.DataFrame(rows).dropna(subset=['expected'])


def zscore(q):
    """Observed minus expected dorsal count over its Poisson-binomial SD."""
    return (q.up - q.expected).sum() / np.sqrt((q.expected * (1 - q.expected)).sum())


def coherence(D, seed=0):
    """Mean pairwise label correlation of the wired cells against matched sets."""
    rng = np.random.default_rng(seed)
    coh = lambda L: np.nanmean(np.corrcoef(L)[np.triu_indices(len(L), 1)])
    out = []
    for (mo, dy), g in D.groupby(['mouse', 'day']):
        z = load_session_labels(int(mo), int(dy))
        if z is None:
            continue
        lab = {int(c): z['labels'][i] for i, c in enumerate(z['cluster_id'])}
        g = g[[c in lab and np.isfinite(lab[c]).all() and np.std(lab[c]) > 0
               for c in g.cell]].copy()
        if g.member.sum() < 4 or (~g.member).sum() < 2:
            continue
        # spike-count terciles within identity, the strata the matched sets keep
        g['q'] = g.groupby('ident').lns.transform(
            lambda x: pd.qcut(x.rank(method='first'), min(3, len(x)), labels=False))
        mem = g[g.member]
        obs = coh(np.array([lab[c] for c in mem.cell]))
        null = []
        for _ in range(N_NULL):
            pick = []
            for _, r in mem.iterrows():
                pool = g[(g.ident == r.ident) & (g.q == r.q) & ~g.cell.isin(pick)]
                if not len(pool):
                    pool = g[(g.ident == r.ident) & ~g.cell.isin(pick)]
                pick.append(int(pool.cell.values[rng.integers(len(pool))]))
            null.append(coh(np.array([lab[c] for c in pick])))
        null = np.array(null)
        out.append(dict(mouse=mo, day=dy, n_wired=len(mem), n_cells=len(g), obs=obs,
                        null=null.mean(), null_sd=null.std(), pct=float((null < obs).mean())))
    return pd.DataFrame(out)


def letter(ax, s, dx=-.2, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom', ha='right')


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=6.5, length=2.5)


def main():
    ID, CN, K, R = tables()
    cpath, hpath = f'{MS}/subnetwork_cells.csv', f'{MS}/subnetwork_coherence.csv'
    D = cell_table(ID, CN, K, R)
    D.to_csv(cpath, index=False)
    O = pd.read_csv(hpath) if os.path.exists(hpath) else coherence(D)
    O.to_csv(hpath, index=False)
    d = D.dropna(subset=['excess'])
    print(f'{len(D)} MEC grid cells and interneurons in {D.sess.nunique()} sessions, '
          f'{int(D.member.sum())} wired; {len(d)} switching')

    dpath = f'{MS}/connection_direction.csv'
    if not os.path.exists(dpath) or 'a0' not in pd.read_csv(dpath, nrows=1):
        direction_table(CN, K).to_csv(dpath, index=False)
    V = pd.read_csv(dpath)

    fig = plt.figure(figsize=(10.2, 9.4))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.35, 1, 1, 1], height_ratios=[1, 1, .95],
                          hspace=.62, wspace=.55, left=.06, right=.98, top=.95, bottom=.06)
    gs_low = gs[2, :].subgridspec(1, 4, width_ratios=[1, 1, 1.5, 1], wspace=.55)

    # A: the example network on the probe
    ax = fig.add_subplot(gs[:2, 0])
    mo, dy = EXAMPLE
    g = D[(D.mouse == mo) & (D.day == dy)].set_index('cell')
    k = K[(K.mouse == mo) & (K.day == dy)]
    # the 4-shank Neuropixels 2.0 probe (probeinterface NP2014), as Figure 8 draws
    # it: every pad light, the session's 384 recorded pads darker
    import probeinterface as pi
    from probeinterface.plotting import plot_probe
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    probe = pi.get_probe('imec', 'NP2014')
    A_ = pd.read_csv(f'/Volumes/datasetsSD/Ephys_data/clark2025/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv')
    on = set(zip(A_.coord_probe_x.round().astype(int), A_.coord_probe_y.round().astype(int)))
    xy = np.round(probe.contact_positions).astype(int)
    act = np.array([(x_, y_) in on for x_, y_ in xy])
    plot_probe(probe, ax=ax, contacts_kargs=dict(alpha=0),
               probe_shape_kwargs=dict(facecolor='0.97', edgecolor='0.6', lw=.4, alpha=1), title=False)
    for sel, col in ((~act, '0.88'), (act, '0.68')):
        ax.add_collection(PatchCollection([Rectangle((x_ - 6, y_ - 6), 12, 12) for x_, y_ in xy[sel]],
                                          facecolor=col, lw=0, zorder=1.2))
    ya = xy[act, 1]
    ax.set_aspect('auto')
    ax.set_xlim(-60, 3 * 250 + 32 + 60); ax.set_ylim(ya.min() - 170, ya.max() + 290)
    ax.scatter(k.px, k.py, s=4, color='0.15', lw=0, zorder=1.5)
    cc = CN[(CN.mouse == mo) & (CN.day == dy)]
    cc = cc[cc.pre.isin(g.index) & cc.post.isin(g.index)]
    jit = {c: (np.random.default_rng(int(c)).uniform(-9, 9)) for c in g.index}
    X = lambda c: g.px[c] + jit[c]
    for a, b, kind in zip(cc.pre.astype(int), cc.post.astype(int), cc.kind):
        ax.annotate('', (X(b), g.py[b]), (X(a), g.py[a]),
                    arrowprops=dict(arrowstyle='-|>', lw=.6, color=INH_C if kind == 'inh' else EXC_C,
                                    alpha=.7, shrinkA=3, shrinkB=3, mutation_scale=5), zorder=2)
    vl = float(np.nanpercentile(np.abs(d.excess), 97))
    norm = TwoSlopeNorm(0, -vl, vl)
    for idn, mk in (('grid', '^'), ('putative interneuron', 'o')):
        q = g[g.ident == idn]
        f = q.excess.notna()
        ax.scatter([X(c) for c in q.index[f]], q.py[f], c=q.excess[f], cmap=AGREE_CMAP, norm=norm,
                   s=34, marker=mk, edgecolor=ICOL[idn], lw=1.1, zorder=3)
        # locked cells have no agreement: hatched grey, not white (white is chance)
        ax.scatter([X(c) for c in q.index[~f]], q.py[~f], color='0.8', hatch='//////',
                   s=34, marker=mk, edgecolor=ICOL[idn], lw=1.1, zorder=3)
    ax.set_xticks([16 + 250 * k_ for k_ in range(4)]); ax.set_xticklabels(['1', '2', '3', '4'])
    ax.set_xlabel('shank', fontsize=7); ax.set_ylabel('µm from probe tip', fontsize=7)
    style(ax)
    ax.set_title(f'M{mo} D{dy}: {int(g.member.sum())} of {len(g)} wired', fontsize=7.5, loc='left')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='^', ls='', mfc='w', mec=ICOL['grid'], label='grid'),
                       Line2D([], [], marker='o', ls='', mfc='w', mec=ICOL['putative interneuron'],
                              label='interneuron'),
                       Line2D([], [], color=INH_C, lw=1, label='inhibitory'),
                       Line2D([], [], color=EXC_C, lw=1, label='excitatory'),
                       Line2D([], [], marker='o', ls='', mfc='0.8', mec='0.5', label='locked'),
                       Line2D([], [], marker='o', ls='', mfc='0.15', mec='none', ms=3, label='other cells'),
                       Line2D([], [], marker='s', ls='', mfc='0.68', mec='none', label='recorded contact'),
                       Line2D([], [], marker='s', ls='', mfc='0.88', mec='0.7', mew=.4, label='inactive contact')],
              fontsize=5.8, frameon=False, loc='upper left', bbox_to_anchor=(0, .99), ncol=2,
              handlelength=1.2, columnspacing=.8)
    cax = ax.inset_axes([.62, .03, .3, .018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm, AGREE_CMAP), cax=cax, orientation='horizontal')
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cax.set_title('agreement above chance', fontsize=5.5, pad=2)
    letter(ax, 'A', dx=-.12, dy=1.01)

    # B: identity-level co-following
    ax = fig.add_subplot(gs[0, 1])
    P = pd.read_csv(f'{MS}/pairs_by_identity.csv')
    S = P.groupby(['mouse', 'day', 'pair']).r.mean().unstack()
    keys = [('grid+grid', ICOL['grid']), ('grid+int', '#7a5aa8'),
            ('int+int', ICOL['putative interneuron'])]
    for i, (kk, col) in enumerate(keys):
        q = (S[kk] - S.other).dropna()
        ax.scatter(i + np.random.default_rng(i).uniform(-.15, .15, len(q)), q, s=7, color=col,
                   alpha=.6, lw=0)
        ax.errorbar(i, q.mean(), q.std(ddof=1) / np.sqrt(len(q)), color='k', marker='_', ms=12,
                    capsize=2.5, lw=1.2)
        p = wilcoxon(q).pvalue
        ax.text(i, ax.get_ylim()[1] if i else q.max(), '', fontsize=1)
        ax.annotate(f'p={p:.1g}\nn={len(q)}', (i, 1.0), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=5.8, color='0.3')
        print(f'B {kk}: {q.mean():+.3f} over other pairs, p = {p:.2g}, {len(q)} sessions')
    ax.axhline(0, color='0.6', lw=.7)
    ax.set_xticks(range(3)); ax.set_xticklabels(['grid-\ngrid', 'grid-\nint', 'int-\nint'], fontsize=6.5)
    ax.set_ylabel('pair label r minus\nother pairs (per session)', fontsize=7)
    style(ax); letter(ax, 'B')

    # C: pooled, wired cells look like stronger followers
    ax = fig.add_subplot(gs[0, 2])
    for i, idn in enumerate(GI):
        for j, (mem, fc) in enumerate(((False, 'w'), (True, ICOL[idn]))):
            v = d[(d.ident == idn) & (d.member == mem)].excess
            x = i * 2.6 + j
            ax.bar(x, v.mean(), .8, color=fc, edgecolor=ICOL[idn], lw=1)
            ax.errorbar(x, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color='k', lw=.9, capsize=2)
            ax.text(x, v.mean() + v.std(ddof=1) / np.sqrt(len(v)) + .006, f'{len(v)}',
                    ha='center', va='bottom', fontsize=5.5, color='0.35')
    ax.set_xticks([.5, 3.1]); ax.set_xticklabels(['grid', 'interneuron'], fontsize=6.8)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('agreement above chance', fontsize=7)
    ax.set_title('pooled: open unwired, filled wired', fontsize=6.6, loc='left', color='0.3')
    style(ax); letter(ax, 'C')

    # D: within session, no difference
    ax = fig.add_subplot(gs[0, 3])
    res = {}
    for i, idn in enumerate(GI):
        q = d[d.ident == idn]
        w = q.groupby(['sess', 'member']).excess.mean().unstack().dropna()
        dd = w[True] - w[False]
        ax.scatter(i + np.random.default_rng(i).uniform(-.15, .15, len(dd)), dd, s=7,
                   color=ICOL[idn], alpha=.6, lw=0)
        ax.errorbar(i, dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd)), color='k', marker='_', ms=12,
                    capsize=2.5, lw=1.2)
        m = smf.ols('excess ~ member + lns + C(sess)', q).fit()
        b, p = m.params['member[T.True]'], m.pvalues['member[T.True]']
        lo, hi = m.conf_int().loc['member[T.True]']
        res[idn] = (b, p, lo, hi, len(dd))
        ax.annotate(f'{b:+.3f}\np={p:.2f}', (i, 1.0), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=5.8, color='0.3')
        print(f'D {idn}: wired - unwired {b:+.3f} [{lo:+.3f}, {hi:+.3f}], p = {p:.2g} '
              f'(session FE + log spikes); {len(dd)} sessions with both, paired mean {dd.mean():+.3f}')
    ax.axhline(0, color='0.6', lw=.7)
    ax.set_xlim(-.6, 1.6); ax.set_xticks([0, 1]); ax.set_xticklabels(['grid', 'interneuron'], fontsize=6.8)
    ax.set_ylabel('wired minus unwired\n(per session)', fontsize=7)
    style(ax); letter(ax, 'D')

    # E: degree against agreement, residualised
    ax = fig.add_subplot(gs[1, 1:3])
    d = d.copy()
    d['res_e'] = smf.ols('excess ~ lns + C(sess) + C(ident)', d).fit().resid
    d['ldeg'] = np.log1p(d.deg)
    d['res_d'] = smf.ols('ldeg ~ lns + C(sess) + C(ident)', d).fit().resid
    for idn in GI:
        q = d[d.ident == idn]
        ax.scatter(q.res_d, q.res_e, s=7, color=ICOL[idn], alpha=.55, lw=0, label=idn.replace('putative ', ''))
    rho, p = spearmanr(d.res_d, d.res_e)
    m = smf.ols('excess ~ ldeg + lns + C(sess) + C(ident)', d).fit()
    xs = np.linspace(d.res_d.min(), d.res_d.max(), 2)
    ax.plot(xs, m.params['ldeg'] * xs, color='k', lw=1)
    ax.set_xlabel('log(1 + grid/interneuron partners), residual', fontsize=7)
    ax.set_ylabel('agreement, residual', fontsize=7)
    ax.set_title(f'rho = {rho:+.3f}, p = {p:.2f}  (residuals on session and log spike count; '
                 f'{len(d)} cells)', fontsize=6.6, loc='left', color='0.3')
    ax.legend(fontsize=6, frameon=False, loc='upper right', markerscale=1.5)
    style(ax); letter(ax, 'E', dx=-.1)
    print(f'E partners vs agreement: rho {rho:+.3f}, p = {p:.2g}; model coef {m.params["ldeg"]:+.3f} '
          f'p = {m.pvalues["ldeg"]:.2g}')

    # F: coherence of the wired set against matched sets
    ax = fig.add_subplot(gs[1, 3])
    ax.errorbar(O.null, O.obs, xerr=O.null_sd, fmt='o', ms=3.2, color='#7a5aa8', ecolor='0.75',
                elinewidth=.7, capsize=0, mec='none')
    lim = [min(O.null.min(), O.obs.min()) - .03, max(O.null.max(), O.obs.max()) + .03]
    ax.plot(lim, lim, color='0.6', lw=.7, ls='--'); ax.set_xlim(lim); ax.set_ylim(lim)
    p = wilcoxon(O.obs - O.null).pvalue
    ax.set_xlabel('matched sets (mean +- SD)', fontsize=7)
    ax.set_ylabel('wired cells', fontsize=7)
    ax.set_title(f'mean pair label r\n{(O.obs - O.null).mean():+.3f}, p = {p:.2f}, {len(O)} sessions',
                 fontsize=6.6, loc='left', color='0.3')
    style(ax); letter(ax, 'F')
    print(f'F coherence wired minus matched {(O.obs - O.null).mean():+.4f}, p = {p:.2g}, '
          f'{len(O)} sessions; median percentile {O.pct.median():.2f}')

    # G: angle of each connection in the probe plane, relative to the position-matched expectation
    edges_a = np.linspace(0, 2 * np.pi, N_ANG + 1)
    mid = .5 * (edges_a[:-1] + edges_a[1:])
    acols = [f'a{i}' for i in range(N_ANG)]
    for j, kind in enumerate(('inh', 'exc')):
        ax = fig.add_subplot(gs_low[0, j], projection='polar')
        q = V[V.kind == kind]
        obs = np.histogram(np.arctan2(q.dy_um, q.dx_um) % (2 * np.pi), edges_a)[0] / len(q)
        exp = q[acols].sum().values / len(q)
        cnt = obs * len(q)
        ratio, se = obs / exp, np.sqrt(cnt) / (exp * len(q))
        ok = exp * len(q) >= 30                   # few connections run across shanks; fade those
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(th, np.ones_like(th), color='k', lw=.8, ls='--')
        ax.plot(np.r_[mid, mid[0]], np.r_[ratio, ratio[0]], color=KIND_C[kind], lw=1)
        for f, al in ((ok, 1), (~ok, .3)):
            ax.errorbar(mid[f], ratio[f], se[f], fmt='o', ms=2.8, color=KIND_C[kind], alpha=al,
                        elinewidth=.8, capsize=0, mec='none')
        ax.set_rlim(0, 1.6); ax.set_rticks([.5, 1, 1.5])
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(['', 'dorsal', '', 'ventral'], fontsize=6.3)
        ax.tick_params(axis='x', pad=-2); ax.tick_params(axis='y', labelsize=5, pad=0)
        ax.set_rlabel_position(45)
        ax.grid(lw=.4, color='0.85'); ax.spines['polar'].set_lw(.5)
        name = 'inhibitory' if kind == 'inh' else 'excitatory'
        ax.set_title(f'{name}, {len(q)}\n{q.up.mean():.1%} dorsal, {q.expected.mean():.1%} expected\n'
                     f'(pooled z = {zscore(q):.1f})', fontsize=6.3, color='0.3', pad=10)
        if j == 0:
            letter(ax, 'G', dx=-.08, dy=1.1)
            ax.text(1.25, -.2, 'observed / expected from cell positions, per direction of post from pre '
                    '(dashed: chance; +-1 SE; faded: < 30 expected)', transform=ax.transAxes,
                    ha='center', fontsize=5.8, color='0.3')
        print(f'G {kind}: {len(q)} MEC connections, {q.up.mean():.3f} dorsal vs {q.expected.mean():.3f} '
              f'expected, z = {zscore(q):.2f}')

    # H: observed minus expected by distance, both kinds
    ax = fig.add_subplot(gs_low[0, 2])
    edges = [0, 50, 100, 200, 400, 1000]
    for j, kind in enumerate(('inh', 'exc')):
        q = V[V.kind == kind].assign(b=lambda x: pd.cut(x.dist, edges))
        st = q.groupby('b', observed=True).apply(lambda g: pd.Series(dict(
            m=(g.up - g.expected).mean(), se=np.sqrt((g.expected * (1 - g.expected)).sum()) / len(g),
            n=len(g))), include_groups=False)
        x = np.arange(len(st)) + (j - .5) * .18
        ax.errorbar(x, st.m, 1.96 * st.se, color=KIND_C[kind], marker='o', ms=3.5, lw=1, capsize=2,
                    label='inhibitory' if kind == 'inh' else 'excitatory')
        for xi, (m, se, n) in zip(x, st[['m', 'se', 'n']].values):
            ax.text(xi, m + 1.96 * se + .01, f'{int(n)}', ha='center', va='bottom', fontsize=5,
                    color=KIND_C[kind])
    ax.axhline(0, color='0.6', lw=.7)
    ax.set_xticks(range(len(edges) - 1))
    ax.set_xticklabels([f'{a}-{b}' for a, b in zip(edges[:-1], edges[1:])], fontsize=6.5)
    ax.set_xlabel('pre-post distance (um)', fontsize=7)
    ax.set_ylabel('fraction pointing dorsal,\nobserved minus expected', fontsize=7)
    ax.set_title('pooled, 95% CI from the per-connection expectation',
                 fontsize=6.6, loc='left', color='0.3')
    ax.legend(fontsize=6, frameon=False, loc='lower left')
    style(ax); letter(ax, 'H', dx=-.16)

    # I: per session, all distances
    ax = fig.add_subplot(gs_low[0, 3])
    for j, kind in enumerate(('inh', 'exc')):
        q = V[V.kind == kind]
        S = q.assign(e=q.up - q.expected).groupby(['mouse', 'day']).agg(e=('e', 'mean'), n=('e', 'size'))
        S = S[S.n >= 10]
        M = S.groupby(level=0).e.mean()
        ax.scatter(j + np.random.default_rng(j).uniform(-.15, .15, len(S)), S.e, s=7,
                   color=KIND_C[kind], alpha=.6, lw=0)
        ax.errorbar(j, S.e.mean(), S.e.std(ddof=1) / np.sqrt(len(S)), color='k', marker='_', ms=12,
                    capsize=2.5, lw=1.2)
        p = wilcoxon(S.e).pvalue
        ax.annotate(f'{S.e.mean():+.3f}\np={p:.2g}\n{len(S)} sess', (j, 1.0),
                    xycoords=('data', 'axes fraction'), ha='center', va='bottom', fontsize=5.8, color='0.3')
        print(f'I {kind}: {S.e.mean():+.3f}, {int((S.e > 0).sum())}/{len(S)} sessions dorsal, '
              f'p = {p:.2g}; mice {int((M > 0).sum())}/{len(M)} {M.round(3).to_dict()}')
    ax.axhline(0, color='0.6', lw=.7)
    ax.set_xlim(-.6, 1.6); ax.set_xticks([0, 1])
    ax.set_xticklabels(['inhibitory', 'excitatory'], fontsize=6.8)
    ax.set_ylabel('dorsal, observed minus expected\n(per session, >= 10 connections)', fontsize=7)
    style(ax); letter(ax, 'I')

    fig.savefig(OUT, dpi=220, bbox_inches='tight')
    print('wrote', OUT)


if __name__ == '__main__':
    main()

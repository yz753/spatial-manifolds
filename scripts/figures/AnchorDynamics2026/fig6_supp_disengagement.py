"""Supplement to Figure 6 -- the pupil when the animal is completely disengaged.

Usage:  python3 fig6_supp_disengagement.py
        (needs data/eye_anchoring/pupil_disengagement{.csv,_profiles.npz},
         from pupil_disengagement.py)

Figure 6 measures the pupil only while the animal is running over the task
track, so it cannot say whether the pupil is wider when the animal is doing
nothing at all -- the prediction if the anchored state sits at the optimum of
an inverted-U arousal curve. This figure adds the epochs Figure 6 leaves out,
with complete disengagement defined two ways: stopped without licking, and
runs of trials in which no reward is earned on either trial type.

    A   pupil against track position, one line per epoch class. The track is
        not uniformly lit -- the first 30 cm of every lap is a black box -- so
        this panel is also the reason every contrast beside it is taken within
        position bins
    B   position-matched contrasts against running: stopped while licking,
        stopped without licking, and long stops without licking
    C   the anchored-minus-non-anchored contrast of Figure 6, recomputed
        within running frames and within disengaged frames
    D   no-hit blocks (>= 3 consecutive trials with no reward of either type,
        including cued trials) against the rest of the session, running
        frames only

POSITION MATCHING. Each contrast is the mean within-10-cm-bin difference over
the bins holding at least 30 frames of both classes. Stops are not spread
evenly over the track -- they cluster in the reward zone -- so an unmatched
contrast is partly a luminance contrast, with the raw difference and the
matched one coming out opposite signs in some sessions. B, C and D also drop
the black box, where a stationary pupil has longer to dark-adapt than a
moving one, which position matching alone does not remove.

STATISTICS. One point per session, Wilcoxon signed-rank against zero, and the
same test with mouse as a random effect (intercept-only mixed model). Where a
contrast is null the minimum detectable effect at 80% power, 2.8 SD / sqrt(n),
is given, so that a null reads as a bound rather than an absence.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
sys.path.insert(0, f'{ROOT}/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR      # noqa: E402

plt.rcParams['font.family'] = 'Arial'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
EA = f'{ROOT}/data/eye_anchoring'
CSV, NPZ = f'{EA}/pupil_disengagement.csv', f'{EA}/pupil_disengagement_profiles.npz'

RUN_C = '#2c7fb8'          # running
LICK_C = '#78c679'         # stopped, licking: still in the task
DIS_C = '#d7301f'          # stopped, not licking: disengaged
BLACK_BOX, RZ = 30.0, (90.0, 110.0)
MIN_BINS = 3               # a session needs this many matched position bins


def _lp(ax, s, dx=-.22, dy=1.0):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def _style(ax):
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)


def stats(v, mouse):
    """Median, Wilcoxon against zero, the same with mouse as a random effect,
    and the minimum detectable effect at 80% power."""
    v = np.asarray(v, float)
    k = np.isfinite(v)
    v, mo = v[k], np.asarray(mouse)[k]
    n = len(v)
    if n < 5:
        return dict(n=n, med=np.nan, p=np.nan, p_me=np.nan, mde=np.nan)
    p = wilcoxon(v)[1]
    try:
        d = pd.DataFrame(dict(y=v, mouse=mo))
        p_me = smf.mixedlm('y ~ 1', d, groups=d.mouse).fit(reml=False).pvalues['Intercept']
    except Exception:
        p_me = np.nan
    return dict(n=n, med=float(np.median(v)), p=float(p), p_me=float(p_me),
                mde=float(2.8 * np.std(v, ddof=1) / np.sqrt(n)),
                n_mice=int(len(set(mo.tolist()))))


def fmt_p(p):
    if not np.isfinite(p):
        return 'n/a'
    return 'p < 1e-4' if p < 1e-4 else f'p = {p:.3g}'


def paired_strip(ax, series, labels, colors, ylab, title, rng):
    """One column of session points per contrast, median bar, zero line."""
    for i, (v, c) in enumerate(zip(series, colors)):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        x = i + (rng.random(len(v)) - .5) * .28
        ax.scatter(x, v, s=7, color=c, alpha=.55, linewidths=0, zorder=2)
        if len(v):
            ax.plot([i - .27, i + .27], [np.median(v)] * 2, color=c, lw=2, zorder=3)
    ax.axhline(0, color='0.55', lw=.8, zorder=0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(title, fontsize=8)
    _style(ax)


def main():
    if not (os.path.exists(CSV) and os.path.exists(NPZ)):
        raise SystemExit(f'missing {CSV}; run pupil_disengagement.py first')
    df = pd.read_csv(CSV)
    Z = np.load(NPZ, allow_pickle=True)
    P, names, posx = Z['profiles'], list(Z['names']), Z['pos']

    # Figure 6's own session filter, carried through by pupil_disengagement.py,
    # so this supplement describes exactly the sessions Figure 6 describes
    inc = df.paper_inc.astype(bool)
    df, P = df[inc].reset_index(drop=True), P[inc.to_numpy()]
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(10.0, 3.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1.1, 1.45, .75],
                          wspace=.55, left=.07, right=.995, top=.86, bottom=.30)

    # ---- A: pupil against track position, by epoch class -------------------
    ax = fig.add_subplot(gs[0, 0]); _lp(ax, 'A', dx=-.17)
    for nm, c, lab in [('moving', RUN_C, 'Running'),
                       ('stopped_lick', LICK_C, 'Stopped, licking'),
                       ('stopped_nolick', DIS_C, 'Stopped, not licking')]:
        A = P[:, names.index(nm), :]
        n = np.sum(np.isfinite(A), axis=0)
        m = np.nanmean(A, axis=0)
        se = np.nanstd(A, axis=0) / np.sqrt(np.maximum(n, 1))
        ok = n >= 5
        ax.plot(posx[ok], m[ok], color=c, lw=1.4,
                label=f'{lab} ({int(np.nanmax(n))})', zorder=3)
        ax.fill_between(posx[ok], (m - se)[ok], (m + se)[ok], color=c, alpha=.22,
                        linewidth=0, edgecolor='none', zorder=2)
    y0, y1 = ax.get_ylim()
    ax.add_patch(Rectangle((0, y0), BLACK_BOX, y1 - y0, color='0.88', zorder=0, lw=0))
    ax.add_patch(Rectangle((200 - BLACK_BOX, y0), BLACK_BOX, y1 - y0, color='0.88',
                           zorder=0, lw=0))
    ax.add_patch(Rectangle((RZ[0], y0), RZ[1] - RZ[0], y1 - y0, color='#9ecae1',
                           alpha=.35, zorder=0, lw=0))
    ax.set_ylim(y0, y1); ax.set_xlim(0, 200)
    ax.set_xlabel('Track position (cm)', fontsize=8)
    ax.set_ylabel('Pupil radius (z)', fontsize=8)
    ax.set_title('Pupil varies with position: the track is not evenly lit',
                 fontsize=8)
    ax.legend(fontsize=6.2, frameon=False, loc='upper left', handlelength=1.2)
    ax.text(.015, .03, 'shaded: black box (grey), reward zone (blue)',
            transform=ax.transAxes, fontsize=5.8, color='0.35')
    _style(ax)

    # ---- B: disengaged vs running, position-matched ------------------------
    ax = fig.add_subplot(gs[0, 1]); _lp(ax, 'B')
    cols = ['d_eng_move', 'd_dis_move_lit', 'd_dislong_move']
    nbs = ['nb_eng_move', 'nb_dis_move_lit', 'nb_dislong_move']
    labs = ['Stopped,\nlicking', 'Stopped,\nnot licking', 'Long stops,\nnot licking']
    ser = [df[c].where(df[n] >= MIN_BINS) for c, n in zip(cols, nbs)]
    paired_strip(ax, ser, labs, [LICK_C, DIS_C, DIS_C],
                 'Pupil minus running (z)', 'Position-matched, lit track', rng)
    # the effect this measure DOES detect, at this sample size, as a yardstick
    ref = stats(df.d_state_move.where(df.nb_state_move >= MIN_BINS), df.mouse)
    for sgn in (-1, 1):
        ax.axhline(sgn * abs(ref['med']), color=ANCH_COLOR, lw=.8, ls='--', zorder=1)
    txt = [f'dashed: the Figure 6 state difference, {abs(ref["med"]):.2f} z']
    for lab, s in zip(['licking', 'not licking', 'long'], ser):
        st = stats(s, df.mouse)
        txt.append(f'{lab}: {st["med"]:+.3f}, {fmt_p(st["p"])}'
                   + (f', MDE {st["mde"]:.2f}' if st['p'] > .05 else ''))
    ax.text(.5, -.30, '\n'.join(txt), transform=ax.transAxes, fontsize=5.6,
            ha='center', va='top', color='0.25', linespacing=1.4)

    # ---- C: no-hit blocks --------------------------------------------------
    ax = fig.add_subplot(gs[0, 2]); _lp(ax, 'C')
    ser = [df.d_nohit_vs_hit.where(df.nb_nohit_vs_hit >= MIN_BINS),
           df.d_nohit.where(df.nb_nohit >= MIN_BINS),
           df.d_nohit_in_anch.where(df.nb_nohit_in_anch >= MIN_BINS),
           df.d_nohit_in_non.where(df.nb_nohit_in_non >= MIN_BINS)]
    paired_strip(ax, ser, ['vs\nrewarded', 'vs rest', 'vs rest,\nanchored',
                           'vs rest,\nnon-anch.'],
                 ['#6a51a3', '#9e9ac8', ANCH_COLOR, NONANCH_COLOR],
                 'No-hit blocks minus\ncomparison (z)',
                 'Blocks with no reward earned', rng)
    txt = []
    for lab, sr in zip(['vs rewarded', 'vs rest', 'within anchored',
                        'within non-anchored'], ser):
        st = stats(sr, df.mouse)
        txt.append(f'{lab}: {st["med"]:+.3f}, {fmt_p(st["p"])}, '
                   f'mouse RE {fmt_p(st["p_me"])}')
    ax.text(.5, -.30, '\n'.join(txt), transform=ax.transAxes, fontsize=5.6,
            ha='center', va='top', color='0.25', linespacing=1.4)

    # ---- D: those blocks are the non-anchored state ------------------------
    ax = fig.add_subplot(gs[0, 3]); _lp(ax, 'D', dx=-.34)
    k = np.isfinite(df.frac_anch_in_nohit)
    a_in, a_all = df.frac_anch_in_nohit[k].to_numpy(), df.frac_anch[k].to_numpy()
    for x, y in zip(a_all, a_in):
        ax.plot([0, 1], [x, y], color='0.75', lw=.6, zorder=1)
    ax.scatter(np.zeros(k.sum()) + (rng.random(k.sum()) - .5) * .1, a_all, s=7,
               color=ANCH_COLOR, alpha=.6, linewidths=0, zorder=2)
    ax.scatter(np.ones(k.sum()) + (rng.random(k.sum()) - .5) * .1, a_in, s=7,
               color='#6a51a3', alpha=.6, linewidths=0, zorder=2)
    for i, v in enumerate([a_all, a_in]):
        ax.plot([i - .22, i + .22], [np.median(v)] * 2,
                color=[ANCH_COLOR, '#6a51a3'][i], lw=2, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Whole\nsession', 'No-hit\nblocks'],
                                              fontsize=7)
    ax.set_xlim(-.4, 1.4); ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction of trials anchored', fontsize=8)
    ax.set_title('What those blocks are', fontsize=8)
    _style(ax)
    pw = wilcoxon(a_in, a_all)[1]
    d_ = pd.DataFrame(dict(y=a_in - a_all, mouse=df.mouse[k].to_numpy()))
    try:
        pm = smf.mixedlm('y ~ 1', d_, groups=d_.mouse).fit(reml=False).pvalues['Intercept']
    except Exception:
        pm = np.nan
    ax.text(.5, -.30, f'{np.median(a_all):.2f} to {np.median(a_in):.2f}, '
            f'{fmt_p(pw)},\nmouse RE {fmt_p(pm)} ({k.sum()} sessions)',
            transform=ax.transAxes, fontsize=5.6, ha='center', va='top',
            color='0.25', linespacing=1.4)

    out = f'{FIG}/fig6_supp_disengagement.pdf'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print('saved', out)

    # ---- the numbers, for the caption and the reply ------------------------
    print(f'\n{len(df)} sessions, {df.mouse.nunique()} mice')
    print(f'time budget: running {df.frac_moving.mean():.2f}, '
          f'stopped+licking {df.frac_eng.mean():.2f}, '
          f'stopped+not licking {df.frac_dis.mean():.2f}')
    print(f'stop duration: not licking {df.dur_dis.mean():.1f} s, '
          f'licking {df.dur_eng.mean():.1f} s')
    print(f'disengaged stops in the reward zone: {df.frac_dis_in_rz.mean():.2f}; '
          f'licking stops: {df.frac_eng_in_rz.mean():.2f}')
    for nm, c, nb in [('stopped+licking - running', 'd_eng_move', 'nb_eng_move'),
                      ('disengaged - running (lit)', 'd_dis_move_lit', 'nb_dis_move_lit'),
                      ('long disengaged - running', 'd_dislong_move', 'nb_dislong_move'),
                      ('disengaged - running, UNMATCHED', 'd_dis_move', 'nb_dis_move'),
                      ('disengaged - licking stops', 'd_dis_eng', 'nb_dis_eng'),
                      ('state, running', 'd_state_move', 'nb_state_move'),
                      ('state, disengaged', 'd_state_dis', 'nb_state_dis'),
                      ('state, disengaged (no RZ)', 'd_state_dis_norz', 'nb_state_dis_norz'),
                      ('state, licking stops', 'd_state_eng', 'nb_state_eng'),
                      ('disengaged - running, anchored', 'd_dis_move_a', 'nb_dis_move_a'),
                      ('disengaged - running, non-anchored', 'd_dis_move_n', 'nb_dis_move_n'),
                      ('no-hit blocks - rest', 'd_nohit', 'nb_nohit'),
                      ('no-hit - rest, within non-anchored', 'd_nohit_in_non', 'nb_nohit_in_non'),
                      ('no-hit - rest, within anchored', 'd_nohit_in_anch', 'nb_nohit_in_anch'),
                      ('no-hit blocks - hit trials', 'd_nohit_vs_hit', 'nb_nohit_vs_hit')]:
        st = stats(df[c].where(df[nb] >= MIN_BINS), df.mouse)
        print(f'  {nm:38s} {st["med"]:+.3f}  {fmt_p(st["p"]):>10s}  '
              f'mouseRE {fmt_p(st["p_me"]):>10s}  MDE {st["mde"]:.3f}  n={st["n"]}')
    print(f'\nno-hit blocks: {df.n_nohit_trials.sum()} trials over '
          f'{(df.n_nohit_trials > 0).sum()} sessions, '
          f'{df.frac_nohit.mean():.2f} of trials; '
          f'anchored fraction inside them {df.frac_anch_in_nohit.mean():.2f} '
          f'against {df.frac_anch.mean():.2f} overall')
    print('raw (unmatched) means: running %.3f, licking stops %.3f, disengaged %.3f'
          % (df.z_move.mean(), df.z_eng.mean(), df.z_dis.mean()))


if __name__ == '__main__':
    main()

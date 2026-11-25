"""Pupil radius when the animal is STATIONARY, and in the black box.

Written to answer a reviewer-style question: the paper's pupil measure
(`figure5_pupil_arousal.ipynb`) is computed only on the moving epoch (speed >
3 cm/s) and only over the task track, so it says nothing about the pupil when
the animal is not running or not doing the task. If the anchored state is an
OPTIMAL point on an inverted-U arousal curve, then quiet, off-task time should
sit BELOW the anchored state's pupil size; if the anchored state is simply low
arousal, quiet time should look the same as it or smaller still.

The comparison has to be luminance-controlled. The virtual track is not
uniformly lit -- the first 30 cm is an unlit black box -- so pupil size varies
with track position for optical reasons that have nothing to do with state.
Stationary epochs are not uniformly distributed over position either (they
cluster in the reward zone), so a raw stationary-vs-moving contrast is partly a
luminance contrast. Everything below is therefore also computed POSITION-MATCHED:
the moving/stationary difference is taken within 10 cm position bins and then
averaged over the bins that have both, which holds luminance fixed by
construction.

Pupil radius, the head-movement correction, the landmark set and the likelihood
cut are all taken from figure5_pupil_arousal.ipynb, so this cannot drift from
the paper's measure. The only thing that changes is the epoch it is evaluated on.

Writes data/eye_anchoring/pupil_stationary.csv, one row per session.
"""
import os, sys, json, glob, re, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import pynapple as nap
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
NBP = f'{ROOT}/scripts/figures/AnchorDynamics2026/figure5_pupil_arousal.ipynb'
_g = {}
for _i, _c in enumerate(json.load(open(NBP))['cells']):
    if _c['cell_type'] == 'code' and _i in (1, 6):      # setup, head_movement
        exec(compile(''.join(_c['source']), '<pupil_nb>', 'exec'), _g)
for _k in ('SOURCE_PATH', 'CLS', 'TL', 'BS', 'NBIN', 'RZ', 'MIN_CELLS',
           'vr_paths', 'clip_trials', 'LM', 'LIK_MIN', 'head_movement'):
    globals()[_k] = _g[_k]

OUT = f'{ROOT}/data/eye_anchoring/pupil_stationary.csv'
STATE = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
BLACK_BOX = 30.0           # cm; unlit, start of every lap
SPEED_THR = 3.0            # cm/s, the paper's moving threshold
MIN_STAT_S = 1.0           # a stationary epoch must last this long to count
LONG_STAT_S = 5.0          # "quiet", clearly off-task
POS_BIN = 10.0             # cm, for the position-matched contrast


def runs(mask, t, min_s):
    """Index arrays of contiguous True runs of `mask` lasting >= min_s."""
    m = np.asarray(mask); out = []
    if m.sum() == 0:
        return out
    edge = np.diff(np.r_[0, m.view(np.int8), 0])
    for a, b in zip(np.where(edge == 1)[0], np.where(edge == -1)[0]):
        if t[b - 1] - t[a] >= min_s:
            out.append(np.arange(a, b))
    return out


def session(mouse, day):
    bp, cp = vr_paths(mouse, day)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    beh = nap.load_file(bp)
    ks = set(beh.keys())
    if not all(f'eye_{k}_{c}' in ks for k in LM for c in ('x', 'y', 'likelihood')):
        return None
    st = STATE[(STATE.mouse == mouse) & (STATE.day == day)]
    if not len(st):
        return None
    # TRIAL NUMBERING. The NWB's trial_number counts every trial; the state
    # table counts only those clip_trials keeps (fully inside the ephys
    # window), renumbered from 1. The two differ by the trials dropped at the
    # start -- 21 of 245 in M20 D14 -- so the lookup goes through the original
    # numbers clip_trials returns. (Fixed 2026-11-20; before that the frames of
    # each trial were given another trial's state.)
    trials_all = beh['trials'].as_dataframe()
    _tr, _orig = clip_trials(trials_all, nap.load_file(cp))
    to_clipped = {int(o): int(n) for o, n in zip(_orig, _tr.number.astype(int))}
    # bool(), not the numpy bool_: `np.True_ is True` is False, which silently
    # left every frame unlabelled.
    anch = {int(k): bool(v) for k, v in zip(st.trial, st.frac_anch > 0.5)}

    # pupil radius at camera rate, same definition as the paper
    lik = np.nanmean([np.asarray(beh[f'eye_{k}_likelihood'].values) for k in LM], axis=0)
    X = np.array([np.asarray(beh[f'eye_{k}_x'].values, float) for k in LM])
    Y = np.array([np.asarray(beh[f'eye_{k}_y'].values, float) for k in LM])
    cx, cy = X.mean(axis=0), Y.mean(axis=0)
    rad = np.hypot(X - cx, Y - cy).mean(axis=0)
    t = np.asarray(beh['eye_x'].index, float)
    fps = 1.0 / np.median(np.diff(t))
    bad = lik < LIK_MIN
    cxm, cym, radm = cx.copy(), cy.copy(), rad.copy()
    cxm[bad] = cym[bad] = radm[bad] = np.nan
    hm_bad, _, _, _, _ = head_movement(cxm, cym, radm, fps=fps)
    bad = bad | hm_bad
    r = rad.copy(); r[bad] = np.nan

    # speed, track position and trial number at camera times
    def at(key):
        s = beh[key]
        return np.interp(t, np.asarray(s.index, float), np.asarray(s.values, float))
    spd = at('S')
    pos = at('P')
    trn = np.round(at('trial_number')).astype(int)

    good = np.isfinite(r)
    if good.sum() < 1000:
        return None
    mu, sd = np.nanmean(r), np.nanstd(r)
    z = (r - mu) / sd if sd > 0 else r * np.nan

    trn = np.array([to_clipped.get(int(n), -1) for n in trn])
    state = np.array([anch.get(int(n), None) for n in trn], dtype=object)
    is_a = np.array([v is True for v in state])
    is_n = np.array([v is False for v in state])
    labelled = is_a | is_n

    moving = spd > SPEED_THR
    stat_m = np.zeros(len(t), bool)
    for ix in runs(spd <= SPEED_THR, t, MIN_STAT_S):
        stat_m[ix] = True
    stat_long = np.zeros(len(t), bool)
    for ix in runs(spd <= SPEED_THR, t, LONG_STAT_S):
        stat_long[ix] = True
    in_bb = pos < BLACK_BOX
    in_rz = (pos >= RZ[0]) & (pos <= RZ[1])

    def m(sel, v=z):
        sel = sel & good
        return float(np.nanmean(v[sel])) if sel.sum() >= 30 else np.nan

    def n(sel):
        return int((sel & good).sum())

    # position-matched moving vs stationary: within-bin difference, then averaged
    def matched(sel_a, sel_b, v=z, lo=0.0, drop_rz=False):
        """Mean within-position-bin difference, over bins with >=30 frames of each.

        `lo` drops bins below a track position, used to exclude the unlit black
        box, where a stationary pupil has time to dilate fully to the dark and a
        moving one does not -- an adaptation difference that position matching
        alone does not remove.
        """
        b = np.floor(pos / POS_BIN).astype(int)
        keep = pos >= lo
        if drop_rz:
            keep = keep & ~in_rz
        ds = []
        for k in np.unique(b[keep]):
            ka, kb = (b == k) & sel_a & good & keep, (b == k) & sel_b & good & keep
            if ka.sum() >= 30 and kb.sum() >= 30:
                ds.append(np.nanmean(v[ka]) - np.nanmean(v[kb]))
        return (float(np.mean(ds)), len(ds)) if ds else (np.nan, 0)

    d_sm, nb_sm = matched(stat_m, moving)
    d_sm_long, nb_sml = matched(stat_long, moving)
    d_sm_lit, nb_lit = matched(stat_m, moving, lo=BLACK_BOX)
    d_sm_long_lit, nb_lit_l = matched(stat_long, moving, lo=BLACK_BOX)
    d_an, nb_an = matched(moving & is_a, moving & is_n)
    d_an_stat, nb_ans = matched(stat_m & is_a, stat_m & is_n)
    # the reward zone is where anchored trials earn reward and non-anchored ones
    # do not, and reward consumption itself dilates the pupil, so the stationary
    # state contrast is also taken with the reward zone dropped entirely
    d_an_stat_norz, nb_ansn = matched(stat_m & is_a, stat_m & is_n,
                                      lo=BLACK_BOX, drop_rz=True)
    d_an_move_norz, nb_amn = matched(moving & is_a, moving & is_n,
                                     lo=BLACK_BOX, drop_rz=True)
    # stop duration by state: a longer stop has longer for the pupil to adapt
    def stopdur(sel):
        ds = [t[ix[-1]] - t[ix[0]] for ix in runs(spd <= SPEED_THR, t, MIN_STAT_S)
              if sel[ix].mean() > 0.5]
        return (float(np.mean(ds)), len(ds)) if ds else (np.nan, 0)
    sd_a, n_sd_a = stopdur(is_a)
    sd_n, n_sd_n = stopdur(is_n)

    # Only the first second of each stop. Stops are longer in the non-anchored
    # state, and a longer stop gives the pupil longer to adapt, so the stationary
    # state contrast is repeated on a window where that cannot differ.
    early = np.zeros(len(t), bool)
    for ix in runs(spd <= SPEED_THR, t, MIN_STAT_S):
        early[ix[t[ix] - t[ix[0]] <= 1.0]] = True
    d_an_early, nb_ae = matched(early & is_a, early & is_n,
                                lo=BLACK_BOX, drop_rz=True)

    # The ordering an inverted-U account needs: quiet time BELOW the anchored
    # state, which is below the non-anchored state. Each running state is
    # compared with quiet time at the SAME track positions.
    d_a_vs_quiet, nb_aq = matched(moving & is_a, stat_long, lo=BLACK_BOX, drop_rz=True)
    d_n_vs_quiet, nb_nq = matched(moving & is_n, stat_long, lo=BLACK_BOX, drop_rz=True)

    return dict(
        mouse=mouse, day=day, fps=round(fps, 1), n_frames=int(good.sum()),
        rad_med=float(np.nanmedian(r)), frac_bad=float(np.mean(bad)),
        frac_stat=float(np.mean(spd <= SPEED_THR)),
        frac_stat_run=float(stat_m.mean()), frac_stat_long=float(stat_long.mean()),
        frac_stat_in_rz=float((stat_m & in_rz).sum() / max(stat_m.sum(), 1)),
        n_trials=int(len(st)), frac_anch=float((st.frac_anch > .5).mean()),
        # raw, for magnitude
        raw_move=m(moving, r), raw_stat=m(stat_m, r), raw_stat_long=m(stat_long, r),
        # z, by epoch
        z_move=m(moving), z_stat=m(stat_m), z_stat_long=m(stat_long),
        z_stat_out=m(stat_m & ~in_rz), z_stat_long_out=m(stat_long & ~in_rz),
        z_stat_lit=m(stat_m & ~in_rz & ~in_bb),
        z_stat_long_lit=m(stat_long & ~in_rz & ~in_bb),
        z_move_lit=m(moving & ~in_rz & ~in_bb),
        z_bb_move=m(moving & in_bb), z_bb_stat=m(stat_m & in_bb),
        # z, by state
        z_move_a=m(moving & is_a), z_move_n=m(moving & is_n),
        z_stat_a=m(stat_m & is_a), z_stat_n=m(stat_m & is_n),
        z_stat_long_a=m(stat_long & is_a), z_stat_long_n=m(stat_long & is_n),
        z_bb_a=m(in_bb & is_a), z_bb_n=m(in_bb & is_n),
        z_bb_move_a=m(moving & in_bb & is_a), z_bb_move_n=m(moving & in_bb & is_n),
        # position-matched contrasts
        d_stat_move=d_sm, nbin_stat_move=nb_sm,
        d_stat_move_lit=d_sm_lit, nbin_stat_move_lit=nb_lit,
        d_stat_long_move_lit=d_sm_long_lit, nbin_stat_long_move_lit=nb_lit_l,
        d_stat_long_move=d_sm_long, nbin_stat_long_move=nb_sml,
        d_anch_move=d_an, nbin_anch_move=nb_an,
        d_anch_stat=d_an_stat, nbin_anch_stat=nb_ans,
        d_anch_stat_norz=d_an_stat_norz, nbin_anch_stat_norz=nb_ansn,
        d_anch_move_norz=d_an_move_norz, nbin_anch_move_norz=nb_amn,
        stopdur_a=sd_a, stopdur_n=sd_n, n_stop_a=n_sd_a, n_stop_n=n_sd_n,
        d_anch_stat_early=d_an_early, nbin_anch_stat_early=nb_ae,
        d_anch_vs_quiet=d_a_vs_quiet, nbin_anch_vs_quiet=nb_aq,
        d_non_vs_quiet=d_n_vs_quiet, nbin_non_vs_quiet=nb_nq,
        n_stat=n(stat_m), n_stat_long=n(stat_long),
        n_move_a=n(moving & is_a), n_move_n=n(moving & is_n),
        n_stat_a=n(stat_m & is_a), n_stat_n=n(stat_m & is_n),
        frac_labelled=float(np.mean(labelled)))


if __name__ == '__main__':
    rows, t0 = [], time.time()
    ds = sorted(glob.glob(f'{SOURCE_PATH}M*/D*/VR'))
    for i, d in enumerate(ds, 1):
        mo, dy = [int(x) for x in re.search(r'M(\d+)/D(\d+)/VR', d).groups()]
        try:
            r = session(mo, dy)
        except Exception as e:
            print(f'  M{mo} D{dy} FAILED: {type(e).__name__}: {e}', flush=True)
            continue
        if r is None:
            continue
        rows.append(r)
        print(f'[{i}/{len(ds)}] M{mo} D{dy}  stat {r["frac_stat"]:.2f}  '
              f'z_move {r["z_move"]:+.3f}  z_stat {r["z_stat"]:+.3f}  '
              f'matched {r["d_stat_move"]:+.3f}  ({time.time()-t0:.0f}s)', flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'\nwrote {OUT}: {len(rows)} sessions in {time.time()-t0:.0f}s')

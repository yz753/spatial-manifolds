"""Is the pupil wider when the animal is COMPLETELY disengaged from the task?

Asked by Matt (2026-11-20). The paper's pupil measure
(`figure5_pupil_arousal.ipynb`) is computed only while the animal is running
(speed > 3 cm/s) over the task track, so it says nothing about time when the
animal is doing nothing at all. If the anchored state sits at an optimal point
on an inverted-U arousal curve, complete disengagement should show a WIDER
pupil than either running state; if the state is simply graded arousal,
disengagement should look like the non-anchored state, only more so.

Two definitions of complete disengagement, both suggested in the email:

1. STOPPED AND NOT LICKING. A stop in which the animal never licks is not
   reward consumption, not reward anticipation, and not running: it is the
   animal doing nothing. Stops that do contain licks are kept as a separate
   class, because they are stationary but still task-engaged, which separates
   "not moving" from "not participating".

2. NO-HIT BLOCKS. Runs of at least three consecutive trials with no reward
   earned on either trial type, containing at least one cued trial -- the
   animal is missing rewards it could get simply by looking at the cue, which
   is the behavioural signature of being out of the task altogether.

Everything is position-matched and state-split, which is what makes the
comparison interpretable:

  LUMINANCE. The virtual track is not uniformly lit: the first 30 cm of every
  lap is an unlit black box, and the pupil tracks that for optical reasons
  with nothing to do with engagement. Stops are not uniformly distributed over
  position either -- they cluster in the reward zone -- so a raw disengaged-vs-
  running contrast is partly a luminance contrast. Every contrast below is
  therefore taken WITHIN 10 cm position bins and averaged over the bins that
  hold enough frames of both classes, which holds luminance fixed by
  construction. A lit-track-only variant drops the black box as well, where a
  stationary pupil has longer to dark-adapt than a moving one -- an adaptation
  difference that position matching alone does not remove.

  ANCHORING STATE. Anchored and non-anchored trials already differ in pupil
  size (Figure 6), so every contrast is also computed separately within each
  state, and the state contrast is recomputed within disengaged time.

Pupil radius, the head-movement correction, the landmark set and the
likelihood cut are taken from figure5_pupil_arousal.ipynb by exec'ing its
cells, so this cannot drift from the paper's measure. Only the epochs change.

Writes data/eye_anchoring/pupil_disengagement.csv (one row per session) and
pupil_disengagement_profiles.npz (per-session pupil against track position,
one profile per epoch class, for the figure's first panel).
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

OUT = f'{ROOT}/data/eye_anchoring/pupil_disengagement.csv'
OUT_NPZ = f'{ROOT}/data/eye_anchoring/pupil_disengagement_profiles.npz'

# The state comes from the pupil figure's OWN per-trial table, not from
# population_state/anchoring_trials.csv: figure5_pupil_arousal.ipynb computes
# labels for every session with eye tracking, and 8 of its 29 included
# sessions have no row in anchoring_trials.csv, which is built on a stricter
# cell-count rule. Taking the state from here makes this supplement describe
# exactly Figure 6's sessions. Both tables number trials the same way (after
# clip_trials), so the mapping below applies to either.
STATE = pd.read_csv(f'{ROOT}/data/eye_anchoring/eye_anchoring_trials.csv')


def _paper_inclusion(D):
    """Figure 6's session filter, applied to its own per-trial table."""
    D = D.copy()
    D['ok'] = D.coverage > 0.5
    cls = D.groupby(['mouse', 'day']).agg(
        n_anch=('frac_anch', lambda s: (s > 0.5).sum()),
        n_tot=('frac_anch', 'size')).reset_index()
    cls['minority'] = np.minimum(cls.n_anch, cls.n_tot - cls.n_anch)
    g = D.groupby(['mouse', 'day']).agg(
        blink=('blink_frac', 'first'), rad_sd=('rad_raw', 'std'),
        rad_med=('rad_raw', 'median'), n_ok=('ok', 'sum')).reset_index()
    g = g.merge(cls[['mouse', 'day', 'minority']], on=['mouse', 'day'])
    g['cv'] = g.rad_sd / g.rad_med
    g['paper_inc'] = ((g.blink <= .20) & (g.n_ok >= 40) & (g.cv >= .015) &
                      (g.minority >= 30))
    return g[['mouse', 'day', 'paper_inc', 'minority']]


INC = _paper_inclusion(STATE)

BLACK_BOX = 30.0        # cm; unlit, start of every lap
SPEED_THR = 3.0         # cm/s, the paper's moving threshold
MIN_STAT_S = 1.0        # a stop must last this long to count as a stop
LONG_STAT_S = 5.0       # a long stop: clearly not a brief hesitation
LICK_PAD_S = 0.5        # a lick this close to a stop still counts as licking in it
POS_BIN = 10.0          # cm, the position-matching bin
NPB = int(TL / POS_BIN)
MIN_FR = 30             # frames of each class a bin needs to contribute
NOHIT_RUN = 3           # trials; a no-hit block is at least this long


def runs(mask, t, min_s):
    """Index arrays of contiguous True runs of `mask` lasting >= min_s."""
    m = np.asarray(mask)
    if m.sum() == 0:
        return []
    out = []
    edge = np.diff(np.r_[0, m.view(np.int8), 0])
    for a, b in zip(np.where(edge == 1)[0], np.where(edge == -1)[0]):
        if t[b - 1] - t[a] >= min_s:
            out.append(np.arange(a, b))
    return out


def nohit_block_trials(tr):
    """Trial numbers inside runs of >= NOHIT_RUN consecutive non-hit trials that
    contain at least one CUED trial -- rewards missed that the cue alone would
    have given, which is the email's "miss cued as well as uncued"."""
    tr = tr.sort_values('number')
    hit = (tr.performance == 'hit').to_numpy()
    cued = (tr.type == 'b').to_numpy()
    num = tr.number.to_numpy().astype(int)
    out, i, n = [], 0, len(tr)
    while i < n:
        if hit[i]:
            i += 1
            continue
        j = i
        while j < n and not hit[j]:
            j += 1
        if (j - i) >= NOHIT_RUN and cued[i:j].any():
            out.extend(num[i:j].tolist())
        i = j
    return set(out)


def session(mouse, day):
    bp, cp = vr_paths(mouse, day)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    beh = nap.load_file(bp)
    ks = set(beh.keys())
    if 'lick' not in ks:
        return None
    if not all(f'eye_{k}_{c}' in ks for k in LM for c in ('x', 'y', 'likelihood')):
        return None
    st = STATE[(STATE.mouse == mouse) & (STATE.day == day)]
    if not len(st):
        return None
    _inc = INC[(INC.mouse == mouse) & (INC.day == day)]
    # TRIAL NUMBERING. The NWB's trial_number channel counts every trial in the
    # session; the state and eye tables count only the trials that survive
    # clip_trials (those fully inside the ephys window), renumbered from 1. The
    # two differ by however many trials were dropped at the start -- 21 of 245
    # in M20 D14 -- so the state has to be looked up through the original
    # numbers clip_trials hands back, not through trial_number directly.
    trials_all = beh['trials'].as_dataframe()
    tr, orig = clip_trials(trials_all, nap.load_file(cp))
    to_clipped = {int(o): int(n) for o, n in zip(orig, tr.number.astype(int))}
    # bool(), not the numpy bool_: `np.True_ is True` is False, which would
    # silently leave every frame unlabelled.
    anch = {int(k): bool(v) for k, v in zip(st.trial, st.frac_anch > 0.5)}

    # --- pupil radius at camera rate, exactly the paper's definition ---------
    lik = np.nanmean([np.asarray(beh[f'eye_{k}_likelihood'].values) for k in LM], axis=0)
    X = np.array([np.asarray(beh[f'eye_{k}_x'].values, float) for k in LM])
    Y = np.array([np.asarray(beh[f'eye_{k}_y'].values, float) for k in LM])
    cx, cy = X.mean(axis=0), Y.mean(axis=0)
    rad = np.hypot(X - cx, Y - cy).mean(axis=0)
    t = np.asarray(beh['eye_x'].index, float)
    fps = 1.0 / np.median(np.diff(t))
    blink = lik < LIK_MIN
    cxm, cym, radm = cx.copy(), cy.copy(), rad.copy()
    cxm[blink] = cym[blink] = radm[blink] = np.nan
    hm_bad, _, _, _, _ = head_movement(cxm, cym, radm, fps=fps)
    bad = blink | hm_bad
    r = rad.copy(); r[bad] = np.nan
    good = np.isfinite(r)
    if good.sum() < 1000:
        return None
    mu, sd = np.nanmean(r), np.nanstd(r)
    if not (sd > 0):
        return None
    z = (r - mu) / sd

    # --- behaviour at camera times -----------------------------------------
    def at(key):
        s = beh[key]
        return np.interp(t, np.asarray(s.index, float), np.asarray(s.values, float))
    spd, pos = at('S'), at('P')
    trn = np.round(at('trial_number')).astype(int)

    lk = beh['lick']
    lt = np.asarray(lk.index, float)[np.asarray(lk.values, float) > 0]

    trn_c = np.array([to_clipped.get(int(n), -1) for n in trn])
    state = np.array([anch.get(int(n), None) for n in trn_c], dtype=object)
    is_a = np.array([v is True for v in state])
    is_n = np.array([v is False for v in state])

    moving = spd > SPEED_THR
    in_bb = pos < BLACK_BOX
    in_rz = (pos >= RZ[0]) & (pos <= RZ[1])

    # --- stops, split by whether the animal licked in them ------------------
    stop_any = np.zeros(len(t), bool)       # every stop >= MIN_STAT_S
    dis = np.zeros(len(t), bool)            # stopped, no lick: disengaged
    eng = np.zeros(len(t), bool)            # stopped, licking: engaged but still
    dis_long = np.zeros(len(t), bool)       # the same, stops >= LONG_STAT_S
    n_stop = n_stop_lick = 0
    durs_dis, durs_eng = [], []
    for ix in runs(spd <= SPEED_THR, t, MIN_STAT_S):
        t0, t1 = t[ix[0]], t[ix[-1]]
        licked = np.any((lt >= t0 - LICK_PAD_S) & (lt <= t1 + LICK_PAD_S))
        stop_any[ix] = True
        n_stop += 1
        if licked:
            eng[ix] = True; n_stop_lick += 1; durs_eng.append(t1 - t0)
        else:
            dis[ix] = True; durs_dis.append(t1 - t0)
            if (t1 - t0) >= LONG_STAT_S:
                dis_long[ix] = True

    # --- no-hit blocks (on the clipped, renumbered trials) ------------------
    nohit = nohit_block_trials(tr)
    in_nohit = np.array([int(n) in nohit for n in trn_c])
    hit_tr = set(tr.number[tr.performance == 'hit'].astype(int))
    in_hit = np.array([int(n) in hit_tr for n in trn_c])

    # --- position-matched contrast -----------------------------------------
    def matched(sel_a, sel_b, lo=0.0, drop_rz=False):
        """Mean within-position-bin difference a - b, over bins holding at least
        MIN_FR frames of each. Holding position fixed holds luminance fixed."""
        b = np.floor(pos / POS_BIN).astype(int)
        keep = (pos >= lo) & good
        if drop_rz:
            keep &= ~in_rz
        ds = []
        for k in np.unique(b[keep]):
            ka, kb = (b == k) & sel_a & keep, (b == k) & sel_b & keep
            if ka.sum() >= MIN_FR and kb.sum() >= MIN_FR:
                ds.append(np.nanmean(z[ka]) - np.nanmean(z[kb]))
        return (float(np.mean(ds)), len(ds)) if ds else (np.nan, 0)

    def prof(sel):
        """Mean z in each 10 cm bin, for the figure's position profiles."""
        b = np.floor(pos / POS_BIN).astype(int)
        out = np.full(NPB, np.nan)
        for k in range(NPB):
            m_ = (b == k) & sel & good
            if m_.sum() >= MIN_FR:
                out[k] = np.nanmean(z[m_])
        return out

    def m(sel):
        sel = sel & good
        return float(np.nanmean(z[sel])) if sel.sum() >= MIN_FR else np.nan

    # the three epoch classes, and the two disengagement contrasts
    d_dis_move, nb1 = matched(dis, moving)
    d_dis_move_lit, nb2 = matched(dis, moving, lo=BLACK_BOX)
    d_dis_eng, nb3 = matched(dis, eng)
    d_eng_move, nb4 = matched(eng, moving)
    d_dislong_move, nb5 = matched(dis_long, moving, lo=BLACK_BOX)

    # disengagement within each anchoring state, and the state contrast within
    # each epoch class: the question is whether disengagement adds anything
    # beyond the state the paper already measures
    d_dis_move_a, nb6 = matched(dis & is_a, moving & is_a, lo=BLACK_BOX)
    d_dis_move_n, nb7 = matched(dis & is_n, moving & is_n, lo=BLACK_BOX)
    d_state_move, nb8 = matched(moving & is_a, moving & is_n, lo=BLACK_BOX, drop_rz=True)
    # The reward zone is dropped from the running contrast because reward
    # consumption itself dilates the pupil, and anchored trials earn reward
    # there while non-anchored ones mostly do not. Disengaged frames are
    # non-licking by construction, so there is no consumption to exclude and
    # the zone is kept -- which matters, since dropping it left only 5
    # sessions with enough matched bins.
    d_state_dis, nb9 = matched(dis & is_a, dis & is_n, lo=BLACK_BOX)
    d_state_dis_norz, nb9b = matched(dis & is_a, dis & is_n, lo=BLACK_BOX, drop_rz=True)
    d_state_eng, nb9c = matched(eng & is_a, eng & is_n, lo=BLACK_BOX, drop_rz=True)

    # no-hit blocks, against the rest of the session, running frames only so
    # that this is not the stop contrast again
    d_nohit, nb10 = matched(moving & in_nohit, moving & ~in_nohit, lo=BLACK_BOX)
    d_nohit_vs_hit, nb11 = matched(moving & in_nohit, moving & in_hit, lo=BLACK_BOX)
    # No-hit blocks are mostly non-anchored, and non-anchored trials already
    # have a wider pupil (Figure 6), so the contrast above could be nothing but
    # the state. Repeating it WITHIN each state asks whether missing rewards
    # adds anything to the state the paper already measures.
    d_nohit_in_non, nb12 = matched(moving & in_nohit & is_n,
                                   moving & ~in_nohit & is_n, lo=BLACK_BOX)
    d_nohit_in_anch, nb13 = matched(moving & in_nohit & is_a,
                                    moving & ~in_nohit & is_a, lo=BLACK_BOX)

    n_tr = len(tr)
    frac_lab = float(np.mean(is_a | is_n))
    return dict(
        frac_labelled=frac_lab,
        mouse=mouse, day=day, fps=round(fps, 1), n_frames=int(good.sum()),
        blink=float(np.mean(blink)), rad_med=float(np.nanmedian(r)),
        rad_cv=float(np.nanstd(r) / np.nanmedian(r)),
        n_trials=int(n_tr), frac_anch=float((st.frac_anch > .5).mean()),
        minority=int(min((st.frac_anch > .5).sum(), (st.frac_anch <= .5).sum())),
        paper_inc=bool(_inc.paper_inc.iloc[0]) if len(_inc) else False,
        # how much of each class there is
        frac_moving=float(moving.mean()), frac_stop=float(stop_any.mean()),
        frac_dis=float(dis.mean()), frac_eng=float(eng.mean()),
        frac_dis_long=float(dis_long.mean()),
        n_stop=n_stop, n_stop_lick=n_stop_lick,
        dur_dis=float(np.mean(durs_dis)) if durs_dis else np.nan,
        dur_eng=float(np.mean(durs_eng)) if durs_eng else np.nan,
        frac_dis_in_rz=float((dis & in_rz).sum() / max(dis.sum(), 1)),
        frac_eng_in_rz=float((eng & in_rz).sum() / max(eng.sum(), 1)),
        # raw means per class (unmatched; for magnitude only)
        z_move=m(moving), z_dis=m(dis), z_eng=m(eng), z_dis_long=m(dis_long),
        z_move_a=m(moving & is_a), z_move_n=m(moving & is_n),
        z_dis_a=m(dis & is_a), z_dis_n=m(dis & is_n),
        # position-matched contrasts: the answers
        d_dis_move=d_dis_move, nb_dis_move=nb1,
        d_dis_move_lit=d_dis_move_lit, nb_dis_move_lit=nb2,
        d_dis_eng=d_dis_eng, nb_dis_eng=nb3,
        d_eng_move=d_eng_move, nb_eng_move=nb4,
        d_dislong_move=d_dislong_move, nb_dislong_move=nb5,
        d_dis_move_a=d_dis_move_a, nb_dis_move_a=nb6,
        d_dis_move_n=d_dis_move_n, nb_dis_move_n=nb7,
        d_state_move=d_state_move, nb_state_move=nb8,
        d_state_dis=d_state_dis, nb_state_dis=nb9,
        d_state_dis_norz=d_state_dis_norz, nb_state_dis_norz=nb9b,
        d_state_eng=d_state_eng, nb_state_eng=nb9c,
        # no-hit blocks
        n_nohit_trials=int(len(nohit)),
        frac_nohit=float(len(nohit) / max(n_tr, 1)),
        frac_anch_in_nohit=float(np.mean([anch.get(int(n), False) for n in nohit]))
        if nohit else np.nan,
        d_nohit=d_nohit, nb_nohit=nb10,
        d_nohit_vs_hit=d_nohit_vs_hit, nb_nohit_vs_hit=nb11,
        d_nohit_in_non=d_nohit_in_non, nb_nohit_in_non=nb12,
        d_nohit_in_anch=d_nohit_in_anch, nb_nohit_in_anch=nb13,
        _prof=np.vstack([prof(moving), prof(dis), prof(eng),
                         prof(moving & is_a), prof(moving & is_n),
                         prof(dis & is_a), prof(dis & is_n)]),
    )


PROF_NAMES = ['moving', 'stopped_nolick', 'stopped_lick',
              'moving_anch', 'moving_non', 'dis_anch', 'dis_non']

if __name__ == '__main__':
    rows, profs, t0 = [], [], time.time()
    ds = sorted(glob.glob(f'{SOURCE_PATH}M*/D*/VR'))
    for i, d in enumerate(ds, 1):
        mo, dy = (int(x) for x in re.search(r'M(\d+)/D(\d+)/VR', d).groups())
        try:
            res = session(mo, dy)
        except Exception as e:
            print(f'[{i}/{len(ds)}] M{mo} D{dy}: {type(e).__name__}: {e}')
            continue
        if res is None:
            continue
        profs.append(res.pop('_prof'))
        rows.append(res)
        print(f'[{i}/{len(ds)}] M{mo} D{dy}: dis {res["frac_dis"]:.3f} '
              f'eng {res["frac_eng"]:.3f}  d_dis_move_lit {res["d_dis_move_lit"]:+.3f}  '
              f'({time.time()-t0:.0f}s)')
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    np.savez_compressed(OUT_NPZ, profiles=np.array(profs),
                        names=np.array(PROF_NAMES),
                        mouse=df.mouse.to_numpy(), day=df.day.to_numpy(),
                        pos=np.arange(NPB) * POS_BIN + POS_BIN / 2)
    print(f'\nwrote {OUT}  ({len(df)} sessions, {time.time()-t0:.0f}s)')
    print(f'wrote {OUT_NPZ}')

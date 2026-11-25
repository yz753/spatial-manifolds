"""Detect head-movement artifacts in the pupil centroid trace.

Two artifacts, needing different treatment:

TRANSIENT SPIKES -- the centroid jumps and comes back within a few frames.
Tracking glitches, partial occlusion by a whisker or the lid. Dropped outright.

SUSTAINED STEPS -- the centroid jumps and STAYS at the new baseline: the head has
shifted in the headplate, so the whole eye translates in the camera frame. Two
signatures are required together, and both matter:

  1. the displacement happens in a single frame (a real head slip is fast), and
  2. the baseline 2 s after differs from the baseline 2 s before.

Requiring (1) is what makes this work. An earlier version tested only (2), which
also fires continuously during smooth gaze drift; accumulating a correction from
those false events inflated one session's gaze range from 8 px to 219 px.

The pupil RADIUS must also be unchanged across the step, which is what separates a
head slip (the eye translates) from a genuine gaze shift with dilation.

A step cannot just be dropped -- everything after it is referenced to a baseline
that no longer applies -- so the cumulative step offset is subtracted, removing
the staircase while leaving real drift intact.

Thresholds scale with each session's own median pupil radius, because camera
distance and zoom differ between sessions (median radius 7.8-12.7 px here), so a
fixed pixel threshold is a different physical displacement in each.

STEP_R was calibrated, not assumed. The whole gaze excursion of a session spans
only ~4-10 px (2.5-97.5 pct), i.e. about one pupil radius, so a threshold of half
a radius sits inside the range of ordinary eye movement: at STEP_R=0.5 the
detector fired 0-29 times per session and the accumulated correction INFLATED one
session's gaze range from 9.7 px to 72 px, which is proof the events were real
gaze rather than slips. At STEP_R=1.0 it fires 0-6 times, removes <0.8% of frames,
and the corrected range matches the raw range (61.4 -> 61.2 px summed over nine
sessions) -- the signature of removing a few genuine discontinuities and nothing
else. A displacement of a full pupil radius within one 33 ms frame is not an eye
movement.
"""
import numpy as np, pandas as pd

SPIKE_R  = 0.8      # transient jump, in units of median pupil radius
STEP_R   = 1.0      # single-frame jump that must also shift the baseline
STEP_WIN = 2.0      # seconds each side over which the new baseline must hold
PAD_S    = 0.5      # seconds blanked either side of a detected step


def head_movement(cx, cy, rad, fps=30.0):
    """Returns (bad, cx_corr, cy_corr, n_spike, n_step)."""
    cx = np.asarray(cx, float).copy(); cy = np.asarray(cy, float).copy()
    rad = np.asarray(rad, float)
    n = len(cx)
    r_med = np.nanmedian(rad)
    if not np.isfinite(r_med) or r_med <= 0:
        return np.ones(n, bool), cx, cy, 0, 0
    t_spike, t_step = SPIKE_R * r_med, STEP_R * r_med
    w = max(3, int(STEP_WIN * fps))
    pad = max(1, int(PAD_S * fps))

    d = np.r_[0.0, np.hypot(np.diff(cx), np.diff(cy))]
    jump = np.where(d > t_step)[0]          # candidate discontinuities

    def med(a, lo, hi):
        s = a[max(0, lo):min(n, hi)]
        return np.nanmedian(s) if np.isfinite(s).sum() >= 3 else np.nan

    bad = np.zeros(n, bool)
    ox = np.zeros(n); oy = np.zeros(n)
    n_step = 0
    last = -10 ** 9
    for i in jump:
        if i - last < w:                    # at most one event per window
            continue
        px, py = med(cx, i - w, i), med(cy, i - w, i)
        qx, qy = med(cx, i + 1, i + 1 + w), med(cy, i + 1, i + 1 + w)
        pr, qr = med(rad, i - w, i), med(rad, i + 1, i + 1 + w)
        if not all(np.isfinite(v) for v in (px, py, qx, qy, pr, qr)):
            continue
        if np.hypot(qx - px, qy - py) > t_step and abs(qr - pr) / r_med < 0.15:
            ox[i + 1:] += qx - px
            oy[i + 1:] += qy - py
            bad[max(0, i - pad):min(n, i + pad + 1)] = True
            n_step += 1
            last = i

    cxc, cyc = cx - ox, cy - oy

    # transient spikes, judged on the CORRECTED trace so a step is not re-counted
    base_x = pd.Series(cxc).rolling(w, center=True, min_periods=w // 3).median().values
    base_y = pd.Series(cyc).rolling(w, center=True, min_periods=w // 3).median().values
    spike = np.hypot(cxc - base_x, cyc - base_y) > t_spike
    bad |= spike
    return bad, cxc, cyc, int(spike.sum()), n_step


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    import pynapple as nap
    LM = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']
    hdr = ('session', 'r_med', 'spike%', 'steps', 'bad%', 'x rng', 'corrected')
    print(f'{hdr[0]:9s} {hdr[1]:>6s} {hdr[2]:>7s} {hdr[3]:>6s} {hdr[4]:>6s}'
          f' {hdr[5]:>7s} {hdr[6]:>10s}')
    for mo, dy in ((20,14),(21,20),(22,35),(25,16),(26,18),(27,17),(28,16),(29,23),(20,23)):
        beh = nap.load_file(f'/Volumes/datasetsSD/Ephys_data/clark2025/M{mo}/D{dy}/VR/'
                            f'sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb')
        X = np.array([np.asarray(beh[f'eye_{k}_x'].values, float) for k in LM])
        Y = np.array([np.asarray(beh[f'eye_{k}_y'].values, float) for k in LM])
        lik = np.nanmean([np.asarray(beh[f'eye_{k}_likelihood'].values) for k in LM], axis=0)
        cx, cy = X.mean(0), Y.mean(0)
        rad = np.hypot(X - cx, Y - cy).mean(0)
        cx[lik <= .5] = np.nan; cy[lik <= .5] = np.nan; rad[lik <= .5] = np.nan
        bad, ccx, ccy, nsp, nst = head_movement(cx, cy, rad)
        f = lambda a: np.nanpercentile(a, 97.5) - np.nanpercentile(a, 2.5)
        print(f'M{mo}D{dy:<5d} {np.nanmedian(rad):6.2f} {100*nsp/len(cx):7.2f} {nst:6d} '
              f'{100*bad.mean():6.2f} {f(cx):7.2f} {f(ccx):10.2f}')

"""Phase precession by anchoring state: pacemaker intact, or pacemaker disrupted?

THE QUESTION. Robinson et al. (2024) silenced medial septal GABAergic neurons and
disrupted BOTH grid periodicity and phase precession. If the non-anchored state
were reduced septal pacemaker drive, precession should be DEGRADED when the
population is non-anchored. If instead what changes is the speed pathway with the
pacemaker intact -- which is what a halved speed-theta gain alongside unchanged
spike-theta phase locking looks like -- precession should be PRESERVED. The two
accounts make opposite predictions and this script tests them.

THE CONFOUND, WHICH IS FATAL IF IGNORED. A non-anchored trial is DEFINED as one
whose rate map does not correspond to the cell's usual track position. Measuring
phase against position-within-a-fixed-field on such trials therefore has to look
degraded -- not because temporal coding broke, but because the spatial reference
is wrong by construction. A naive analysis here would "discover" the classifier's
own definition and report it as a septal result.

Two measurements are therefore made:

  NAIVE     precession against position in the cell's pooled field, by state.
            Reported ONLY so the size of the artifact is on record. A difference
            here means nothing on its own.

  MATCHED   the same measurement restricted to trials on which the cell HAS a
            field -- that trial's rate map must correlate with the cell's own
            template above threshold, applied identically in both states. This
            asks the question that can actually be answered: where a field
            exists, does phase still sweep through it when the population is
            non-anchored?

If MATCHED shows preserved precession, the temporal code survives and only the
spatial registration is lost, which favours an intact pacemaker. If MATCHED shows
degraded precession, that favours the septal account. If too few non-anchored
trials carry a field to make the comparison, the honest answer is that the test
cannot be run -- and that is a real possible outcome here, because Results 2
found non-anchored trials show loss of spatial correspondence rather than a
displaced field.

Circular-linear regression follows Kempter et al. (2012): the slope maximising
the resultant length of (phase - 2*pi*a*x), with the circular-linear correlation
coefficient as the effect size.

Writes data/population_state/phase_precession.csv
"""
import os
import sys
import warnings

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pynapple as nap
from lfp_batch import FS, LFP_ROOT
from spatial_manifolds.anchoring import load_session_labels

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
SOURCE = '/Volumes/datasetsSD/Ephys_data/clark2025'
OUT = f'{ROOT}/data/population_state/phase_precession.csv'
CLS = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')

TL, BS = 200.0, 2.0
NBIN = int(TL / BS)
RUN_SPEED = 3.0
FIELD_FRAC = 0.30          # field = contiguous bins >= 30% of peak
MIN_FIELD_CM, MAX_FIELD_CM = 10.0, 80.0
MIN_SPIKES = 50            # per cell per state
MIN_TRIAL_R = 0.30         # trial-vs-template correlation for "has a field"
MIN_TRIALS_STATE = 5
SLOPES = np.linspace(-2.0, 2.0, 801)   # cycles per field traversal


def vr_paths(mo, dy):
    d = f'{SOURCE}/M{mo}/D{dy}/VR/'
    s = f'sub-M{mo}_ses-D{dy}_typ-VR'
    return d + s + '_beh.nwb', d + s + '_srt-kilosort4_clusters.npz'


def circ_lin(phase, x):
    """Kempter circular-linear regression. Returns slope, phase0, rho, n."""
    if len(phase) < 10:
        return np.nan, np.nan, np.nan, len(phase)
    R = [np.abs(np.mean(np.exp(1j * (phase - 2 * np.pi * a * x)))) for a in SLOPES]
    a = SLOPES[int(np.argmax(R))]
    phi0 = np.angle(np.mean(np.exp(1j * (phase - 2 * np.pi * a * x))))
    th = 2 * np.pi * a * x
    pb = np.angle(np.mean(np.exp(1j * phase)))
    tb = np.angle(np.mean(np.exp(1j * th)))
    num = np.sum(np.sin(phase - pb) * np.sin(th - tb))
    den = np.sqrt(np.sum(np.sin(phase - pb) ** 2) * np.sum(np.sin(th - tb) ** 2))
    rho = num / den if den > 0 else np.nan
    return a, phi0, rho, len(phase)


def main_field(tc):
    """Contiguous bins around the peak that exceed FIELD_FRAC of it."""
    if not np.isfinite(tc).any() or np.nanmax(tc) <= 0:
        return None
    pk = int(np.nanargmax(tc))
    thr = FIELD_FRAC * tc[pk]
    lo = pk
    while lo - 1 >= 0 and tc[lo - 1] >= thr:
        lo -= 1
    hi = pk
    while hi + 1 < len(tc) and tc[hi + 1] >= thr:
        hi += 1
    w = (hi - lo + 1) * BS
    if not (MIN_FIELD_CM <= w <= MAX_FIELD_CM):
        return None
    return lo, hi, pk


def run_session(mo, dy):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    bp, cp = vr_paths(mo, dy)
    if not (os.path.exists(lf) and os.path.exists(bp)):
        return []
    z = load_session_labels(mo, dy)
    if z is None:
        return []
    with h5py.File(lf, 'r') as f:
        if 'processing/ecephys/Processed/theta/data' not in f:
            return []
        names = [n.decode() for n in
                 f['general/extracellular_ephys/electrodes/channel_name'][:]]
        ch2g = {c: i for i, n in enumerate(names) for c in n.split('_')}
        beh = nap.load_file(bp)
        clusters = nap.load_file(cp)
        ec = clusters.metadata['extremum_channel']
        lab_ids = [int(c) for c in z['cluster_id']]
        keep = [c for c in lab_ids
                if c in clusters.index and ch2g.get(str(ec[c]), -1) >= 0]
        if len(keep) < 5:
            return []
        cols = sorted({ch2g[str(ec[c])] for c in keep})
        back = {g: i for i, g in enumerate(cols)}
        PH = f['processing/ecephys/Processed/theta/data'][:, cols].astype(np.float32)

    n = PH.shape[0]
    t = np.arange(n) / FS
    P, S, tn = beh['P'], beh['S'], beh['trial_number']
    pos = np.asarray(P.values)[np.searchsorted(np.asarray(P.index), t).clip(0, len(P) - 1)]
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t).clip(0, len(S) - 1)]
    trn = np.asarray(tn.values)[np.searchsorted(np.asarray(tn.index), t).clip(0, len(tn) - 1)]
    trials = np.asarray(z['trial']).astype(int)
    frac = np.asarray(z['frac_anch'], float)
    pop_state = {tr: (fa > .5) for tr, fa in zip(trials, frac)}
    L = z['labels']
    cellrow = {c: i for i, c in enumerate(lab_ids)}

    run_ok = spd >= RUN_SPEED
    rows = []
    for c in keep:
        sp = np.asarray(clusters[c].index, float)
        i = np.round(sp * FS).astype(int)
        i = i[(i >= 0) & (i < n)]
        i = i[run_ok[i]]
        if len(i) < 2 * MIN_SPIKES:
            continue
        ph = PH[i, back[ch2g[str(ec[c])]]].astype(float)
        ph = (ph + np.pi) % (2 * np.pi) - np.pi
        sp_pos = pos[i]
        sp_tr = trn[i].astype(int)

        # --- per-trial rate maps, pooled template, and the cell's main field --
        rowi = cellrow[c]
        labs = L[rowi]
        tmask = np.isin(sp_tr, trials)
        if tmask.sum() < 2 * MIN_SPIKES:
            continue
        ph, sp_pos, sp_tr = ph[tmask], sp_pos[tmask], sp_tr[tmask]
        # occupancy-normalised map per trial
        tidx = {tr: k for k, tr in enumerate(trials)}
        occ = np.zeros((len(trials), NBIN))
        cnt = np.zeros((len(trials), NBIN))
        pb_all = np.clip((pos[run_ok] / BS).astype(int), 0, NBIN - 1)
        tr_all = trn[run_ok].astype(int)
        for tr, pbv in zip(tr_all, pb_all):
            k = tidx.get(tr)
            if k is not None:
                occ[k, pbv] += 1.0 / FS
        for tr, pv in zip(sp_tr, np.clip((sp_pos / BS).astype(int), 0, NBIN - 1)):
            k = tidx.get(tr)
            if k is not None:
                cnt[k, pv] += 1
        with np.errstate(invalid='ignore', divide='ignore'):
            M = cnt / np.where(occ > 0, occ, np.nan)
        tmpl = np.nanmean(M, axis=0)
        fld = main_field(tmpl)
        if fld is None:
            continue
        lo, hi, pk = fld
        # per-trial correlation with the template -> "does this trial have the field"
        tr_r = np.full(len(trials), np.nan)
        for k in range(len(trials)):
            v = M[k]
            good = np.isfinite(v) & np.isfinite(tmpl)
            if good.sum() > NBIN // 3 and np.nanstd(v[good]) > 0:
                tr_r[k] = np.corrcoef(v[good], tmpl[good])[0, 1]
        has_field = {tr: (r > MIN_TRIAL_R) for tr, r in zip(trials, tr_r)}

        infield = (sp_pos >= lo * BS) & (sp_pos <= (hi + 1) * BS)
        xin = (sp_pos - lo * BS) / ((hi - lo + 1) * BS)
        st = np.array([pop_state.get(int(tr), False) for tr in sp_tr])
        hf = np.array([has_field.get(int(tr), False) for tr in sp_tr])

        rec = dict(mouse=mo, day=dy, cluster_id=c,
                   field_lo=lo * BS, field_hi=(hi + 1) * BS,
                   n_trials_anch=int(sum(pop_state.get(int(tr), False) for tr in trials)),
                   n_trials_non=int(sum(not pop_state.get(int(tr), False) for tr in trials)))
        for tag, sel in (('naive', infield),
                         ('matched', infield & hf)):
            for sname, smask in (('anch', st), ('non', ~st)):
                m = sel & smask
                a, p0, rho, nn = circ_lin(ph[m], xin[m]) if m.sum() >= MIN_SPIKES \
                    else (np.nan, np.nan, np.nan, int(m.sum()))
                rec[f'{tag}_{sname}_slope'] = a
                rec[f'{tag}_{sname}_rho'] = rho
                rec[f'{tag}_{sname}_n'] = nn
        # how many trials of each state actually carry the field
        rec['frac_hasfield_anch'] = float(np.mean([has_field.get(int(tr), False)
                                                   for tr in trials if pop_state.get(int(tr), False)] or [np.nan]))
        rec['frac_hasfield_non'] = float(np.mean([has_field.get(int(tr), False)
                                                  for tr in trials if not pop_state.get(int(tr), False)] or [np.nan]))
        rows.append(rec)
    return rows


def main():
    import glob, re
    sess = []
    for d in sorted(glob.glob(f'{SOURCE}/M*/D*/VR')):
        m = re.search(r'M(\d+)/D(\d+)/VR', d)
        sess.append((int(m.group(1)), int(m.group(2))))
    w = int(sys.argv[1]) if len(sys.argv) > 2 else 0
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    sess = sess[w::nw]
    rows = []
    for k, (mo, dy) in enumerate(sess, 1):
        try:
            r = run_session(mo, dy)
        except Exception as e:
            print(f'[{k}/{len(sess)}] M{mo}D{dy} ! {type(e).__name__}: {e}', flush=True)
            continue
        rows += r
        print(f'[{k}/{len(sess)}] M{mo}D{dy} {len(r)} cells', flush=True)
    if not rows:
        sys.exit('no rows')
    D = pd.DataFrame(rows)
    sfx = '' if nw == 1 else f'_w{w}'
    D.to_csv(OUT.replace('.csv', f'{sfx}.csv'), index=False)
    print(f'\nwrote {OUT.replace(".csv", f"{sfx}.csv")}: {len(D)} cells')


if __name__ == '__main__':
    main()

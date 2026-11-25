"""LFP depth profiles across the visual cortex - MEC boundary.

For a session, every LFP channel (each the mean of 8 neighbouring contacts,
~60 um) is placed at its contacts' mean dorsoventral depth and majority atlas
region (anatomy_M*_D*.csv). Over running samples only (>= 3 cm/s), per channel:

  power     Welch spectrum (2 s windows) relative to 1-200 Hz total, in
            delta 1-4, theta 6-10, slow gamma 30-50, fast gamma 60-100 Hz and
            a high band 200-400 Hz (spiking leaks into it: a multi-unit proxy)
  coh       theta-band (6-10 Hz) magnitude-squared coherence with a reference
            channel in MEC (the MEC channel with most theta power)
  phase     theta phase lag to that reference (circular mean of the
            Hilbert-phase difference on 6-10 Hz filtered signals)

Each profile is fitted with one step (two-segment least squares, each segment
>= 3 channels) and the step depth compared with the atlas VIS-MEC border
(midway between the deepest VIS and shallowest MEC channel).

Writes data/lfp/boundary/M{mo}D{dy}.npz
"""
import os, sys, warnings
import h5py, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.signal import welch, coherence, butter, filtfilt, hilbert
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
LFP = '/Volumes/datasetsSD/Ephys_data/clark2025_lfp'
SRC = '/Volumes/datasetsSD/Ephys_data/clark2025'
OUTD = f'{ROOT}/data/lfp/boundary'
FS = 1000.
BANDS = {'delta': (1, 4), 'theta': (6, 10), 'slow gamma': (30, 50), 'fast gamma': (60, 100), 'high 200-400': (200, 400)}
MAX_S = 600          # s of running used (subsampled evenly), to bound memory/time


def region(a):
    a = str(a)
    return 'MEC' if a.startswith('ENTm') else 'VIS' if a.startswith('VIS') else 'other'


def step_fit(d, v, min_n=3):
    o = np.argsort(d); d, v = d[o], v[o]
    best = (np.inf, None)
    for i in range(min_n, len(d) - min_n + 1):
        sse = ((v[:i] - v[:i].mean()) ** 2).sum() + ((v[i:] - v[i:].mean()) ** 2).sum()
        if sse < best[0]:
            best = (sse, (d[i - 1] + d[i]) / 2, v[:i].mean(), v[i:].mean())
    return best[1:]


def session(mo, dy):
    f = f'{LFP}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    A = pd.read_csv(f'{LFP}/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv').set_index('channel_id')
    with h5py.File(f, 'r') as h:
        names = [n.decode() for n in h['general/extracellular_ephys/electrodes/channel_name'][:]]
        n_t = h['processing/ecephys/LFP/lfp/data'].shape[0]
        beh = nap.load_file(f'{SRC}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb')
        S = beh['S']
        t = np.arange(n_t) / FS
        spd = np.interp(t, np.asarray(S.index), np.asarray(S.values))
        run = spd >= 3
        # running samples in contiguous 2 s chunks, evenly subsampled to MAX_S
        L = 2000
        starts = [i for i in range(0, n_t - L, L) if run[i:i + L].mean() > .9]
        if len(starts) * 2 > MAX_S:
            starts = list(np.array(starts)[np.linspace(0, len(starts) - 1, MAX_S // 2).astype(int)])
        X = np.concatenate([h['processing/ecephys/LFP/lfp/data'][s:s + L, :] for s in starts]).astype(np.float32)
    rows = []
    for j, nm in enumerate(names):
        chs = [c for c in nm.split('_') if c in A.index]
        if not chs:
            continue
        a = A.loc[chs]
        rows.append(dict(j=j, dv=a.coord_SCs_y.mean(), x=a.coord_probe_x.mean(),
                         shank=a.device_contact_id.str.extract(r's(\d+)e')[0].astype(int).mode()[0],
                         region=a.brain_region.map(region).mode()[0], acr=a.brain_region.mode()[0]))
    C = pd.DataFrame(rows)
    # one shank: the one with the most VIS + MEC channels
    sh = C[C.region != 'other'].groupby('shank').size().idxmax()
    C = C[C.shank == sh].sort_values('dv').reset_index(drop=True)
    Xs = X[:, C.j.values]
    f_, P = welch(Xs, fs=FS, nperseg=2000, axis=0)
    tot = P[(f_ >= 1) & (f_ <= 200)].sum(0)
    for b, (lo, hi) in BANDS.items():
        C[b] = P[(f_ >= lo) & (f_ <= hi)].sum(0) / tot
    mec = C.region == 'MEC'
    ref = int(C[mec].theta.idxmax()) if mec.any() else int(C.theta.idxmax())
    ff, coh = coherence(Xs, Xs[:, [ref]], fs=FS, nperseg=2000, axis=0)
    C['theta coherence'] = coh[(ff >= 6) & (ff <= 10)].mean(0)
    bb, aa = butter(3, [6 / 500, 10 / 500], btype='band')
    ph = np.angle(hilbert(filtfilt(bb, aa, Xs, axis=0), axis=0))
    C['theta phase lag (deg)'] = np.degrees(np.angle(np.exp(1j * (ph - ph[:, [ref]])).mean(0)))
    vis_d, mec_d = C.dv[C.region == 'VIS'], C.dv[C.region == 'MEC']
    border = (vis_d.max() + mec_d.min()) / 2 if len(vis_d) and len(mec_d) else np.nan
    steps = {}
    for k in list(BANDS) + ['theta coherence', 'theta phase lag (deg)']:
        steps[k] = step_fit(C.dv.values, C[k].values.astype(float))
    os.makedirs(OUTD, exist_ok=True)
    C.to_csv(f'{OUTD}/M{mo}D{dy}.csv', index=False)
    return C, border, steps, ref, len(starts) * 2


if __name__ == '__main__':
    for mo, dy in [(26, 18), (21, 19)]:
        C, border, steps, ref, sec = session(mo, dy)
        print(f'M{mo}D{dy}: {len(C)} channels on the shank, {sec} s running; atlas border {border:.0f} um')
        print('   regions by depth:', ' '.join(f'{r[0]}' for r in C.region))
        for k, (d, a, b) in steps.items():
            print(f'   {k:22s} step at {d:6.0f} um ({d - border:+5.0f} vs border): {a:.3f} above -> {b:.3f} below')

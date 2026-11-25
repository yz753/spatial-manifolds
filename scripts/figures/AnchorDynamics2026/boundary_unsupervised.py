"""Two atlas-free ways to find the VIS-MEC boundary along one shank.

1. SPECTRAL CLUSTERING OF CHANNELS. Each LFP channel's full log power spectrum
   (1-300 Hz, Welch on 2 s windows of running only) is z-scored per frequency
   across channels and reduced to 3 PCs. Channels are then split (a) by k-means
   with k = 2 and 3, ignoring depth, and (b) into two contiguous depth segments
   minimising within-segment variance in PC space (the multivariate step). If
   the spectra carry the boundary, (a) labels change at one depth and (b) lands
   there.

2. WAVEFORMS. Every sorted cluster except Bombcell 'noise' (all mua, good and
   non-soma units, curated or not), at its extremum channel's depth, described
   by |amplitude|, peak-to-trough duration, trough half-width, repolarisation
   slope, recovery slope, SNR and firing rate (log where skewed). Profiles are
   running medians over 150 um; the multivariate two-segment split is taken on
   the per-cell z-scored features ordered by depth.

Each split is compared with the atlas border (midway between the deepest VIS
and shallowest MEC channel / cluster). Writes data/lfp/boundary/unsup_M{mo}D{dy}.npz
"""
import os, sys, warnings
import h5py, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap

LFP = '/Volumes/datasetsSD/Ephys_data/clark2025_lfp'
SRC = '/Volumes/datasetsSD/Ephys_data/clark2025'
OUTD = '/Users/harryclark/Documents/spatial-manifolds/data/lfp/boundary'
FS = 1000.
WFEAT = [('amplitude', lambda m: np.log10(m.amplitude_median.abs())),
         ('peak-to-trough (ms)', lambda m: m.peak_to_trough_duration * 1e3),
         ('trough half-width (ms)', lambda m: m.trough_half_width * 1e3),
         ('repolarisation slope', lambda m: np.log10(m.repolarization_slope.abs())),
         ('recovery slope', lambda m: np.log10(m.recovery_slope.abs())),
         ('SNR', lambda m: np.log10(m.snr)),
         ('firing rate', lambda m: np.log10(m.firing_rate.clip(lower=.01)))]


def region(a):
    a = str(a)
    return 'MEC' if a.startswith('ENTm') else 'VIS' if a.startswith('VIS') else 'other'


def split2(d, F, min_n=3):
    """Best contiguous two-segment split of rows of F ordered by depth d."""
    o = np.argsort(d); d, F = d[o], F[o]
    best = (np.inf, None)
    for i in range(min_n, len(d) - min_n + 1):
        if d[i] == d[i - 1]:
            continue
        sse = ((F[:i] - F[:i].mean(0)) ** 2).sum() + ((F[i:] - F[i:].mean(0)) ** 2).sum()
        if sse < best[0]:
            best = (sse, (d[i - 1] + d[i]) / 2)
    return best[1]


def spectra(mo, dy):
    f = f'{LFP}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    A = pd.read_csv(f'{LFP}/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv').set_index('channel_id')
    beh = nap.load_file(f'{SRC}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb')
    with h5py.File(f, 'r') as h:
        names = [n.decode() for n in h['general/extracellular_ephys/electrodes/channel_name'][:]]
        dset = h['processing/ecephys/LFP/lfp/data']; n_t = dset.shape[0]
        t = np.arange(n_t) / FS
        spd = np.interp(t, np.asarray(beh['S'].index), np.asarray(beh['S'].values))
        L = 2000
        st = [i for i in range(0, n_t - L, L) if (spd[i:i + L] >= 3).mean() > .9]
        st = list(np.array(st)[np.linspace(0, len(st) - 1, min(len(st), 300)).astype(int)])
        X = np.concatenate([dset[s:s + L, :] for s in st]).astype(np.float32)
    rows = []
    for j, nm in enumerate(names):
        chs = [c for c in nm.split('_') if c in A.index]
        if chs:
            a = A.loc[chs]
            rows.append(dict(j=j, dv=a.coord_SCs_y.mean(),
                             shank=a.device_contact_id.str.extract(r's(\d+)e')[0].astype(int).mode()[0],
                             region=a.brain_region.map(region).mode()[0]))
    C = pd.DataFrame(rows)
    sh = C[C.region != 'other'].groupby('shank').size().idxmax()
    C = C[C.shank == sh].sort_values('dv').reset_index(drop=True)
    f_, P = welch(X[:, C.j.values], fs=FS, nperseg=2000, axis=0)
    keep = (f_ >= 1) & (f_ <= 300) & ~((f_ > 47) & (f_ < 53)) & ~((f_ > 97) & (f_ < 103))
    return C, f_[keep], np.log10(P[keep].T)


def session(mo, dy):
    C, f, LP = spectra(mo, dy)
    Z = (LP - LP.mean(0)) / LP.std(0)
    pcs = PCA(3, random_state=0).fit_transform(Z)
    C['k2'] = KMeans(2, n_init=20, random_state=0).fit_predict(pcs)
    C['k3'] = KMeans(3, n_init=20, random_state=0).fit_predict(pcs)
    C[['pc1', 'pc2', 'pc3']] = pcs
    lfp_split = split2(C.dv.values, pcs)
    vis, mec = C.dv[C.region == 'VIS'], C.dv[C.region == 'MEC']
    border_lfp = (vis.max() + mec.min()) / 2
    # waveforms
    m = nap.load_file(f'{SRC}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_srt-kilosort4_clusters.npz').metadata
    m = m[m.bombcell_label.astype(str) != 'noise'].copy()
    m['region'] = m.brain_region.map(region); m['dv'] = m.coord_SCs_y
    W = pd.DataFrame({k: fn(m) for k, fn in WFEAT}, index=m.index)
    ok = W.notna().all(1) & np.isfinite(W).all(1) & m.dv.notna()
    W, m = W[ok], m[ok]
    Wz = ((W - W.mean()) / W.std()).values
    wf_split = split2(m.dv.values, Wz, min_n=15)
    vv, mm = m.dv[m.region == 'VIS'], m.dv[m.region == 'MEC']
    border_wf = (vv.max() + mm.min()) / 2
    os.makedirs(OUTD, exist_ok=True)
    np.savez_compressed(f'{OUTD}/unsup_M{mo}D{dy}.npz', f=f, LP=LP, chan=C.to_records(index=False),
                        wf=W.values, wf_names=np.array([k for k, _ in WFEAT]), wf_dv=m.dv.values,
                        wf_region=m.region.values.astype(str), wf_label=m.bombcell_label.values.astype(str),
                        lfp_split=lfp_split, wf_split=wf_split, border_lfp=border_lfp, border_wf=border_wf)
    return C, lfp_split, border_lfp, m, W, wf_split, border_wf


if __name__ == '__main__':
    for mo, dy in [(26, 18), (21, 19)]:
        C, ls, bl, m, W, ws, bw = session(mo, dy)
        print(f'M{mo}D{dy}: atlas border {bl:.0f} um')
        print('   region :', ''.join(r[0] for r in C.region))
        print('   k=2    :', ''.join(str(k) for k in C.k2))
        print('   k=3    :', ''.join(str(k) for k in C.k3))
        print(f'   spectral two-segment split {ls:.0f} um ({ls - bl:+.0f})')
        print(f'   waveforms: {len(m)} clusters; two-segment split {ws:.0f} um ({ws - bw:+.0f} vs border {bw:.0f})')
        for k in W.columns:
            print(f'      {k:24s} VIS median {W[k][m.region == "VIS"].median():.3f}  MEC {W[k][m.region == "MEC"].median():.3f}')

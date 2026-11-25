"""Spatial information on the VR track and in the open field, for every cluster.

Question: do visual cells carry spatial information on the track but not in
the open field, where MEC cells carry it in both? If so, open-field spatial
information marks MEC in a way theta alone cannot.

Every cluster with atlas coordinates in MEC or visual cortex (curated or not),
in every session with visual cells. Running-only (VR >= 3 cm/s; OF >= 2.5 cm/s)
spike counts in 50 ms bins against position: 2 cm bins on the 200 cm track,
3 cm square bins in OF1. Skaggs information (bits/spike) from the occupancy-
normalised rate map, tested against N_SHUF circular shifts of the cell's counts
(>= 20 s) against position: significant above the 95th percentile.

Writes data/population_state/region_spatial_info.csv
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
SRC = '/Volumes/datasetsSD/Ephys_data/clark2025'
PS = f'{ROOT}/data/population_state'
OUT = f'{PS}/region_spatial_info.csv'
DT = .05; N_SHUF = 50; MIN_SHIFT_S = 20.; MIN_SPK = 50
CX = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')


def region(a):
    a = str(a)
    return 'MEC' if a.startswith('ENTm') else 'VIS' if a.startswith('VIS') else None


def skaggs(cnt, pb, nb):
    occ = np.bincount(pb, minlength=nb).astype(float)
    spk = np.bincount(pb, weights=cnt, minlength=nb)
    m = occ > 0
    p = occ[m] / occ[m].sum(); r = spk[m] / occ[m]; rbar = (p * r).sum()
    if rbar <= 0:
        return np.nan
    q = r / rbar
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.nansum(p * q * np.log2(np.where(q > 0, q, 1))))


def session_env(path_beh, path_cl, kind, ids, rng):
    beh = nap.load_file(path_beh); cl = nap.load_file(path_cl)
    if kind == 'VR':
        P = beh['P']; S = beh['S']
        t = np.asarray(P.index); t0, t1 = t.min(), t.max()
        edges = np.arange(t0, t1, DT); ctr = edges[:-1] + DT / 2
        i = np.searchsorted(t, ctr).clip(0, len(t) - 1)
        pos, spd = np.asarray(P.values)[i], np.asarray(S.values)[i]
        ok = (spd >= 3) & np.isfinite(pos)
        pb = (np.clip(pos, 0, 199.99) // 2).astype(int); nb = 100
    else:
        X, Y = beh['P_x'], beh['P_y']; S = beh['S']
        t = np.asarray(X.index); t0, t1 = t.min(), t.max()
        edges = np.arange(t0, t1, DT); ctr = edges[:-1] + DT / 2
        i = np.searchsorted(t, ctr).clip(0, len(t) - 1)
        x, y, spd = np.asarray(X.values)[i], np.asarray(Y.values)[i], np.asarray(S.values)[i]
        ok = (spd >= 2.5) & np.isfinite(x) & np.isfinite(y)
        xmin, ymin = np.nanmin(x[ok]), np.nanmin(y[ok])
        xb = ((x - xmin) // 3).astype(int); yb = ((y - ymin) // 3).astype(int)
        nx = int(np.nanmax(xb[ok])) + 1; ny = int(np.nanmax(yb[ok])) + 1
        pb = np.where(ok, xb * ny + yb, 0); nb = nx * ny
    pb = pb[ok]
    smin = int(MIN_SHIFT_S / DT)
    res = {}
    have = set(int(c) for c in cl.index)
    for c in ids:
        if c not in have:
            continue
        cnt = np.histogram(np.asarray(cl[c].index), bins=edges)[0].astype(float)
        if cnt[ok].sum() < MIN_SPK:
            continue
        si = skaggs(cnt[ok], pb, nb)
        nul = []
        for k in rng.integers(smin, len(cnt) - smin, N_SHUF):
            nul.append(skaggs(np.roll(cnt, k)[ok], pb, nb))
        res[c] = (si, float(np.nanpercentile(nul, 95)))
    return res


if __name__ == '__main__':
    R = pd.read_csv(f'{PS}/pc1_by_region.csv')
    sess = sorted({(int(a), int(b)) for a, b in
                   R[R.brain_region.astype(str).str.startswith('VIS')][['mouse', 'day']].values})
    rows, t0 = [], time.time()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT); rows = prev.to_dict('records')
        done = set(zip(prev.mouse, prev.day)); sess = [s for s in sess if s not in done]
    for mo, dy in sess:
        c = CX[(CX.mouse == mo) & (CX.day == dy)].dropna(subset=['coord_SCs_y'])
        c = c.assign(reg=c.brain_region.map(region)).dropna(subset=['reg'])
        ids = [int(x) for x in c.cluster_id]
        rng = np.random.default_rng(mo * 100 + dy)
        out = {}
        for kind, ses in (('VR', 'VR'), ('OF', 'OF1')):
            b = f'{SRC}/M{mo}/D{dy}/{ses}/sub-M{mo}_ses-D{dy}_typ-{ses}_beh.nwb'
            p = f'{SRC}/M{mo}/D{dy}/{ses}/sub-M{mo}_ses-D{dy}_typ-{ses}_srt-kilosort4_clusters.npz'
            if os.path.exists(b) and os.path.exists(p):
                try:
                    out[kind] = session_env(b, p, kind, ids, rng)
                except Exception as e:
                    print(f'  ! M{mo}D{dy} {kind}: {type(e).__name__}: {e}', flush=True)
        for _, r in c.iterrows():
            k = int(r.cluster_id)
            d = dict(mouse=mo, day=dy, cluster_id=k, region=r.reg, cls=r.cell_class_of1, dv=r.coord_SCs_y)
            for kind in ('VR', 'OF'):
                si, thr = out.get(kind, {}).get(k, (np.nan, np.nan))
                d[f'si_{kind}'] = si; d[f'thr_{kind}'] = thr; d[f'sig_{kind}'] = bool(si > thr) if np.isfinite(si) else np.nan
            rows.append(d)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f'  M{mo}D{dy}: {len(c)} cells, {time.time() - t0:.0f}s', flush=True)

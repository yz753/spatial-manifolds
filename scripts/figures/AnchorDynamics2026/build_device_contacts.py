"""Every probe contact of every mouse, in atlas coordinates, from clark2025.

The anatomy figures (3, 4, S5, S15) draw all of a mouse's contacts and fit the
best atlas slice through them. Each clark2025 session folder carries
anatomy_M{mo}_D{dy}.csv, one row per recorded channel with its device contact
id, stereotaxic and CCF coordinates and atlas region. A mouse's contacts are the
union of its sessions' rows, one per device contact id.

This reproduces the table the figures were first built from
(device_contact_id_annotations.csv): all 12,288 contacts, coordinates equal to
1e-12 um and regions identical.

Writes data/device_contact_id_annotations.csv
"""
import glob
import os
import re

import pandas as pd

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
CLARK2025 = '/Volumes/datasetsSD/Ephys_data/clark2025/'
OUT = f'{ROOT}/data/device_contact_id_annotations.csv'

if __name__ == '__main__':
    fs = sorted(glob.glob(f'{CLARK2025}M*/D*/anatomy_M*_D*.csv'))
    if not fs:
        raise SystemExit(f'no anatomy_M*_D*.csv under {CLARK2025}')
    A = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    A = A.drop_duplicates(['mouse', 'device_contact_id']).copy()
    # shank_id numbers each mouse's shanks medial -> lateral (rank of |ML|),
    # which is not always the device's own s0-s3 order (M21 is reversed)
    A['s'] = A.device_contact_id.map(lambda s: int(re.match(r's(\d+)e', s).group(1)))
    ml = A.groupby(['mouse', 's']).coord_SCs_x.mean().abs()
    rank = ml.groupby(level=0).rank(method='first').astype(int) - 1
    A['shank_id'] = [rank[(m, s)] for m, s in zip(A.mouse, A.s)]
    A['Mouse'] = 'M' + A.mouse.astype(str)
    A = A.rename(columns={'coord_probe_x': 'unit_location_x', 'coord_probe_y': 'unit_location_y'})
    cols = ['device_contact_id', 'unit_location_x', 'unit_location_y', 'mouse', 'shank_id',
            'Mouse', 'coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x', 'coord_CCFs_z',
            'coord_CCFs_y', 'coord_CCFs_x', 'brain_region']
    A = A.sort_values(['mouse', 'shank_id', 'unit_location_y', 'unit_location_x'])[cols]
    A.reset_index(drop=True).to_csv(OUT)
    print(f'{len(A)} contacts from {len(fs)} sessions -> {OUT}')

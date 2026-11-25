# Entorhinal grid codes are gated by attention

**This repository is specifically for that paper: its analysis, its figures
and its text.** It is not a general-purpose library. Every number and every
panel in the manuscript comes from a script in here, and the manuscript
itself is in here too, as markdown that the builders set as a PDF.

The paper is referred to throughout as *AnchorDynamics2026*, and all of it
lives in
[`scripts/figures/AnchorDynamics2026/`](scripts/figures/AnchorDynamics2026/).
Code outside that folder is the shared package and the preprocessing that
produced the dataset the paper analyses.

## Where to start

- **Making the figures and the PDFs**:
  [`scripts/figures/AnchorDynamics2026/README.md`](scripts/figures/AnchorDynamics2026/README.md)
  — the data paths, the environment, one command per figure or one command for
  all of them (`rebuild_paper.py figures`), which notebooks are still live,
  and the manuscript builders, which need no data at all.
- **Recomputing everything from the recordings**:
  [`scripts/figures/AnchorDynamics2026/REBUILD.md`](scripts/figures/AnchorDynamics2026/REBUILD.md)
  — `rebuild_paper.py derive`, the caches to clear first, what is deliberately
  not in git, and troubleshooting.
- **The text**: `MANUSCRIPT.md` (title through Discussion) and `PAPER.md` (the
  working document, whose `## Methods` section becomes the STAR Methods), both
  in that folder.

## Where the data is

Everything is on the **INCR-NolanLab** group share on the Edinburgh DataStore.
On a Mac it mounts at `/Volumes/INCR-NolanLab`; on Eddie the same tree is
`/exports/cmvm/datastore/sbms/groups/INCR-NolanLab`. These are the paths the
scripts carry — check them when you mount the share and tell Harry if anything
has moved.

| What | Path | Needed for |
|---|---|---|
| **clark2025** — the processed dataset: 81 VR sessions, each with `VR/`, `OF1/`, `OF2/` as `.nwb`, a spike-sorted `kilosort4.npz` and an `anatomy_M*_D*.csv` | `/Volumes/INCR-NolanLab/ActiveProjects/Wolf/harry/data/clark2025` | every analysis; **not** needed to redraw a figure from the intermediate results committed here |
| **clark2025_lfp** — the same sessions with the raw LFP | `/Volumes/INCR-NolanLab/ActiveProjects/Wolf/harry/data/clark2025_lfp` | Figure 8 and Figures S10, S13, S14, S15; all LFP `derive` steps |
| the eye videos, cropped to the eye (`*_eye_zone.avi`) | `/Volumes/INCR-NolanLab/ActiveProjects/Harry/EphysNeuropixelData/vr` | only a Figure 6 or S17 trial whose frames are not already cached in the repository |
| the DeepLabCut eye crop windows | `/Volumes/INCR-NolanLab/ActiveProjects/Chris/Cohort12/derivatives/labels/deeplabcut/all_eye_crops.csv` | same |
| **Allen mouse atlas, 10 µm** — public, not on the share | `pip install brainglobe-atlasapi` then `brainglobe install -a allen_mouse_10um`, which lands in `~/.brainglobe/allen_mouse_10um_v1.2` (~5 GB) | Figure 4 and Figures S7, S8 |

A local copy is faster than reading over the network, but either works;
`rebuild_paper.py configure` is where you say which path to use. Harry's local
copies of clark2025 and clark2025_lfp are both on one SD card, under
`/Volumes/datasetsSD/Ephys_data/`, along with the atlas tiffs — so on that
machine the card must be mounted for any step that opens a session.
The raw Neuropixels recordings, and the preprocessing that turns them into
clark2025, are a separate pipeline — see [`data/README.md`](data/README.md).

## Environment

The paper scripts need Python 3.11 and the packages in
[`requirements-paper.txt`](requirements-paper.txt):

```
pip install -r requirements-paper.txt
```

The rest of the repository, including the preprocessing, uses
[`uv`](https://docs.astral.sh/uv/):

```
uv sync
```

[`uv.lock`](uv.lock) has the full environment. The shared package is in
[`src/spatial_manifolds/`](src/spatial_manifolds/); other analyses, including
earlier projects, are in [`scripts/`](scripts/).

## Contact

The preprocessing pipeline and the `spatial_manifolds` package were written by
Wolf De Wulf:

| Name | Email address | Website |
| :--- | :------------ | :------ |
| Wolf De Wulf | [wolf.de.wulf@ed.ac.uk](mailto:wolf.de.wulf@ed.ac.uk) | https://www.wolfdewulf.eu |

The paper, its analyses and its figures are Harry Clark's; questions about
anything in `AnchorDynamics2026/` go to him.

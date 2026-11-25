# Entorhinal grid codes are gated by attention

*Making the figures and the paper.* This repository is specifically for this
paper -- its analysis, its figures and its text -- and everything behind it,
referred to throughout as *AnchorDynamics2026*, lives in this folder: the
analysis scripts, the figure scripts, the manuscript text, and the builders
that set it all as a PDF. This file is the practical guide — where the processed data is, and the
command for each figure, either all at once or one at a time.

Two documents, so you know which to read:

- **README.md** (this file): make the figures and the PDFs from the *processed*
  data on the DataStore. This is the normal path.
- **[REBUILD.md](REBUILD.md)**: the deeper guide — recomputing every
  intermediate result from the raw recordings (`derive`), the caches that must
  be cleared first, what is deliberately not in git, and troubleshooting.

---

## 1. Where the data is

Everything below is on the **INCR-NolanLab** group share on the Edinburgh
DataStore. On a Mac it mounts at `/Volumes/INCR-NolanLab`; on Eddie the same
tree is `/exports/cmvm/datastore/sbms/groups/INCR-NolanLab`. These are the
paths the scripts carry — check them against the share when you mount it, and
tell Harry if anything has moved.

| What | Path | Size | Needed for |
|---|---|---|---|
| **clark2025** — the processed dataset: 81 VR sessions, each with `VR/`, `OF1/`, `OF2/` as `.nwb` plus a spike-sorted `kilosort4.npz` and an `anatomy_M*_D*.csv` | `/Volumes/INCR-NolanLab/ActiveProjects/Wolf/harry/data/clark2025` | large | every analysis; **not** needed to redraw a figure from the committed intermediates |
| **clark2025_lfp** — the same sessions with the raw LFP | `/Volumes/INCR-NolanLab/ActiveProjects/Wolf/harry/data/clark2025_lfp` | large | Figure 8 and Figures S10, S13, S14, S15; all LFP `derive` steps |
| the eye videos (already cropped to the eye, `*_eye_zone.avi`) | `/Volumes/INCR-NolanLab/ActiveProjects/Harry/EphysNeuropixelData/vr` | ~50 MB/session | only a Figure 6 or S17 trial whose frames are not already cached in `eye_frames_M*D*_trials.npz` |
| the DeepLabCut eye crop windows | `/Volumes/INCR-NolanLab/ActiveProjects/Chris/Cohort12/derivatives/labels/deeplabcut/all_eye_crops.csv` | small | same |
| **Allen mouse atlas, 10 µm** — public, not on the share | `pip install brainglobe-atlasapi` then `brainglobe install -a allen_mouse_10um`, which lands in `~/.brainglobe/allen_mouse_10um_v1.2` | ~5 GB | Figure 4 and Figures S7, S8 (`fig3_anatomy.py`) |

Harry's local copies of both datasets are on one SD card, at
`/Volumes/datasetsSD/Ephys_data/clark2025` and
`/Volumes/datasetsSD/Ephys_data/clark2025_lfp` (moved there 2026-11-18, which
is also where the brainglobe atlas tiffs are symlinked). That is faster than
reading over the network, but it means **the card has to be mounted for any
step that opens a session**, not just the LFP and anatomy ones. Reading the
DataStore paths directly works too; step 3 is where you say which.

**The processed data is enough.** The raw Neuropixels recordings, and the
preprocessing that turns them into clark2025, are a separate pipeline —
[`data/README.md`](../../../data/README.md) in the repository root covers it.
Nothing in this folder goes back that far.

## 2. Environment

Python 3.11, from the repository root:

```bash
pip install -r requirements-paper.txt
```

(The rest of the repository uses `uv sync`; the paper scripts only need the
list above.) Two extras, both optional: `pip install pyphen` for the Nature
builder's hyphenation, and the Microsoft core fonts for Arial — without them
the figures still build, in a fallback font.

Running a script by hand needs the package importable and a non-interactive
matplotlib, which is what `rebuild_paper.py` sets for each step:

```bash
cd scripts/figures/AnchorDynamics2026
export PYTHONPATH=../../../src:$PYTHONPATH
export MPLBACKEND=Agg
```

Every script writes its PDF into this folder, so run them from here.

## 3. Point the scripts at your data (once)

The scripts carry Harry's paths in their string literals. This rewrites them
to yours, everywhere they occur, and remembers what it did in
`rebuild_paths.json`:

```bash
python rebuild_paper.py configure \
    --clark2025 /Volumes/INCR-NolanLab/ActiveProjects/Wolf/harry/data/clark2025 \
    --lfp       /Volumes/INCR-NolanLab/ActiveProjects/Wolf/harry/data/clark2025_lfp \
    --atlas     ~/.brainglobe/allen_mouse_10um_v1.2
python rebuild_paper.py check          # every line should say ok
```

Add `--dry-run` to the first to see the edits without making them, and
`--skip-lfp` to the second if you do not have the LFP dataset yet.

## 4. Everything in one run

```bash
python rebuild_paper.py figures        # ~6 min; every figure, then the PDFs
```

This redraws all 10 main figures, all 19 supplements, the 4 extra figures and
the graphical abstract from the intermediate results already committed to the
repository, then compiles the PDFs. It needs clark2025_lfp for its five LFP-reading
steps and nothing else; `--skip-lfp` leaves those out and keeps the figure PDFs
already in the folder.

Each step logs to `rebuild_logs/<step>.log`. A failure does not stop the rest,
and the summary at the end names what failed. One step on its own:
`--only fig3_4`. From a step onwards: `--from fig7`.

To recompute the intermediate results from clark2025 instead of trusting the
committed ones, `python rebuild_paper.py derive` — many hours, and
[REBUILD.md](REBUILD.md) lists the caches to clear first and the traps.
`python rebuild_paper.py all` is `derive` then `figures`.

## 5. One figure at a time

Each of these writes the PDF that the paper builder reads. Run from this
folder, with the environment of step 2.

### Main figures

| Figure | Command | Writes | Needs |
|---|---|---|---|
| 1 | `python fig1_composite.py` | `fig1_composite.pdf` | its panels, from `figure1_schematics.ipynb` (step 6) |
| 2 | `python fig1_supp_neighbour_bars.py` | `fig1_supp_neighbour_bars.pdf` | — |
| 3 | `python fig2_single_units.py` | `fig2_single_units.pdf` | — |
| 4 | `python fig3_anatomy.py` | `fig4_subnetwork_anatomy.pdf`, and Figures S7 and S8 | the **atlas** |
| 5 | `python fig4_monosynaptic.py` | `fig4_monosynaptic.pdf` | — |
| 6 | `python fig5_pupil_arousal.py` | `fig5_pupil_arousal_M21D19.pdf` | cached eye frames (or the videos) |
| 7 | `python fig7_supp_vis_examples.py` | `fig7_supp_vis_examples.pdf` | — |
| 8 | `python fig7_theta_gamma.py` | `fig7_theta_gamma.pdf` | the **LFP dataset** |
| 9 | `python fig8_supp_attractor_more.py --sim` | `fig8_supp_attractor_sim.pdf` | — (without `--sim` it draws the two-session version, out of the paper since 2026-11-02) |
| 10 | `python fig8_model.py` | `fig8_model.pdf` | — (a schematic; no data) |

### Supplements

| Figure | Command | Writes | Needs |
|---|---|---|---|
| S1 | `python fig1_supp_classifier.py` | `fig1_supp_classifier.pdf` | ~20 min uncached; simulates its own grid cells |
| S2 | `python fig1_supp_population_state.py` | `fig1_supp_population_state.pdf` | — |
| S3 | `python fig1_supp_rasters.py` | `fig1_supp_population_anchoring_examples.pdf` | — |
| S4 | `python fig1_supp_regions.py` | `fig1_supp_regions.pdf` | — |
| S5 | `python fig1_supp_neighbour_test.py` | `fig1_supp_neighbour_test.pdf` | — |
| S6 | `python fig2_supp_nonanchored_rate.py` | `fig2_supp_nonanchored_rate.pdf` | the panels from `fig2_supp_classifier_steps.py` and `fig2_supp_nonanchored.py` first |
| S7, S8 | `python fig3_anatomy.py` | `fig4_supp_anatomy_more.pdf`, `fig4_supp_factors.pdf` | the **atlas**; S8 also `data/population_state/fig4_factors.json` from `fig4_factors.py` |
| S9 | `python region_state_dimensionality.py` | `fig6_supp_dimensionality.pdf` | — |
| S10 | `python fig7_supp_band_examples.py` | `fig7_supp_band_examples.pdf` | the **LFP dataset** |
| S11 | `python fig7_supp_comodulogram.py` | `fig7_supp_comodulogram.pdf` | — |
| S12 | `python fig7_supp_gamma_locking.py` | `fig7_supp_gamma_locking.pdf` | — |
| S13 | `python speed_tuning_fig.py` | `fig7_supp_speed_tuning.pdf` | the **LFP dataset** (its example cell) |
| S14 | `python fig_supp_lfp_spectra.py` | `fig_supp_lfp_spectra.pdf` | the **LFP dataset** |
| S15 | `python fig_supp_histology_example.py` | `fig_supp_histology_example.pdf` and `..._M20.pdf` | the **LFP dataset**, and the section strips in `data/histology/` |
| S16 | `python fig7_supp_more_examples.py` | `fig7_supp_more_examples.pdf` | — |
| S17 | `python fig6_supp_pupil_examples.py` | `fig6_supp_pupil_examples.pdf` | cached eye frames (or the videos) |
| S18 | `python fig5_supp_subnetwork.py` | `fig5_supp_subnetwork.pdf` | — |
| S19 | `python fig5_supp_narrow_continuum.py` | `fig5_supp_narrow_continuum.pdf` | — |

### Extra analyses PDF

| Figure | Command |
|---|---|
| E1 | `python fig1_supp_of_decomp.py` |
| E2 | `python fig3_supp_laminar.py` |
| E3 | `python fig6_region_examples.py` |
| E4 | `python fig6_supp_region_examples.py` |

The script names lag the figure numbers, because the figures were renumbered
during writing; the authoritative mapping is the `FIGURES` list at the top of
[build_cell_pdf.py](build_cell_pdf.py), which pairs each figure number with
its PDF and its caption. [REBUILD.md](REBUILD.md) records the renumbering
history and what the intermediate results behind each figure are.

## 6. The notebooks

Five notebooks are still part of the pipeline. `rebuild_paper.py` executes
them headless, and you can do the same:

```bash
python -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 <notebook>
```

| Notebook | What it is |
|---|---|
| `lick_raster_by_trial_type.ipynb` | **the shared loader.** Almost every analysis script `exec`s its first cells to get the session paths, the curation, the trial clipping and the binning constants, so that no two scripts can drift apart in how they load a session. Nothing to run on its own — but if a path or a curation rule changes, this is where it changes |
| `figure5_pupil_arousal.ipynb` | the pupil measure and the pupil tables. `fig5_pupil_arousal.py` and `pupil_stationary.py` both `exec` cells of it for the same reason |
| `figure1_schematics.ipynb` | Figure 1's panels, which `fig1_composite.py` then assembles |
| `position_decoding_by_anchoring.ipynb` | position decoding by state (`derive` step `decoding`) |
| `session_validity_pc1_significance.ipynb` | per-session PC1 significance (`derive` step `session_validity`) |
| `../figure_0_classification/create_classification_datasheet_v2.ipynb` | `data/cell_classifications_v2.csv`: open-field cell identity, tuning scores and anatomy per cell (`derive` step `classification`) |

The other notebooks here — `figure2_single_units.ipynb`,
`figure3_anatomy_identity.ipynb`, `figure4_monosynaptic.ipynb`,
`figure6_cross_region_coupling.ipynb`, `figure7_theta_anchoring.ipynb`,
`anchoring_classifier_validation.ipynb` and the rest — are the exploratory
originals, from which the `fig*.py` scripts were written. **They are not what
the paper's figures come from**, and most were last touched in September or
October 2026, before the ISI exclusion and the anatomy rules changed the cell
set. Read them for how an analysis was arrived at; do not quote a number from
one.

## 7. The text and the PDFs

The four builders read markdown and the figure PDFs beside them. They never
redraw a figure and never open clark2025, so this works in a fresh clone with
no data and no atlas:

```bash
python build_cell_pdf.py        # -> AnchorDynamics2026_Cell_mockup.pdf (+ _small)
python build_nature_pdf.py      # -> AnchorDynamics2026_Nature_mockup.pdf (+ _small)
python build_extra_pdf.py       # -> AnchorDynamics2026_Extra_analyses.pdf
python build_docx.py            # -> google_doc/AnchorDynamics2026_manuscript.docx
```

A couple of minutes each, most of it `shrink_pdf.py` resampling the images.
Always run `build_nature_pdf.py` after `build_cell_pdf.py`: it imports the
text parser, the captions and the references from it, so the two stay in step
only if both are rebuilt. The Nature builder also needs `pyphen` and the faces
`extract_nature_fonts.py` extracts once into `fonts/`.

Where each piece of the text lives:

| Piece | File |
|---|---|
| Title, Summary, Introduction, Results, Discussion | `MANUSCRIPT.md` |
| STAR Methods | the `## Methods` section of `PAPER.md` |
| Extra analyses text | `EXTRA.md` |
| Figure captions | the `FIGURES` list in `build_cell_pdf.py` |
| References | the `REFS` dict in `build_cell_pdf.py` — cites are author-year in the markdown, and an unresolved key **fails the build**, so add the entry before the citation |

`PAPER.md` is also the working document: everything above its `## Methods`
heading (title options, the argument point by point, figure drafts) is not in
the PDF.

## 8. If you have less than all of it

| You have | You can |
|---|---|
| the repository only | redraw every figure except the five LFP ones and Figure 4's anatomy; rebuild all four PDFs |
| the repository and the atlas | all of the above plus Figures 4, S7 and S8 |
| the repository, atlas and clark2025_lfp | every figure in the paper |
| all of that plus clark2025 | `derive` as well: recompute every intermediate result from the recordings |

[REBUILD.md](REBUILD.md) has the troubleshooting list — a leftover
`/Users/harryclark/...` path, a `check` that finds no sessions, a missing
3D render, p-values that differ slightly after a `derive`.

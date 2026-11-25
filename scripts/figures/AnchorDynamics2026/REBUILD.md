# Rebuilding the AnchorDynamics2026 figures and paper

This folder holds every script behind the paper *AnchorDynamics2026* (the
mock-up PDF is `AnchorDynamics2026_Cell_mockup.pdf`). This guide explains how
to rebuild the figures and the PDF on your own machine from the shared data.

One script does the work: `rebuild_paper.py`. The steps below are all you
need, in order.

For the shorter path -- the DataStore paths, the command for each figure on
its own, which notebooks are still live, and the PDF builders -- see
[README.md](README.md) in this folder. This file is the deeper guide: every
step of the pipeline, recomputing the intermediate results from the raw
recordings, and what to do when something goes wrong.

---

## 1. What you need

| What | Where it comes from | Needed for |
|---|---|---|
| **clark2025** | DataStore (Harry will give you the path). 81 VR sessions, each with `VR/`, `OF1/`, `OF2/` and an `anatomy_M*_D*.csv`. | everything |
| **clark2025_lfp** | DataStore, alongside clark2025. Same sessions, with the raw LFP. | Figure 8 and Figures S10, S13, S14 and S15 at draw time, plus the LFP steps of `derive` |
| **Allen mouse atlas, 10 µm** | Public. Install with `pip install brainglobe-atlasapi` then `brainglobe install -a allen_mouse_10um`. It lands in `~/.brainglobe/allen_mouse_10um_v1.2` (about 5 GB). | the anatomy figures (Figure 4 and its supplements) and the histology supplement |
| **This repository** | `git clone https://github.com/HDClark94/spatial-manifolds.git`, branch `spatial` | everything |
| **Python 3.11** with the packages in `requirements-paper.txt` (in the repository root) | `pip install -r requirements-paper.txt` | everything |

Disk: the repository is a few GB because it carries the intermediate results.
Copying clark2025 locally is faster than reading it over the network, but
either works. On Harry's machine both datasets and the atlas now live on one
SD card under `/Volumes/datasetsSD/` (clark2025 moved there 2026-11-18, off a
nearly full boot volume), so every step that opens a session needs the card
mounted.

### Things that are *not* in clark2025, and are in the repository instead

Almost everything is computed from clark2025 and clark2025_lfp. A few inputs
cannot be, so they are committed to the repository. `rebuild_paper.py check`
confirms they are there:

- **Hand-drawn schematics**: `fig1_schem.pdf` (Figure 1) and
  `camera_and_IR_mouse.pdf` and `pupilometry.pdf` (Figure 7).
- **Eye-video stills** for Figure 7 (`eye_frames_M21D19_trials.npz`). The pupil
  *measurements* come from clark2025; only these example frames come from raw
  videos, which are not part of the dataset.
- **First-pass anatomy table** `data/cell_classifications.csv`, which the
  classification step reads alongside clark2025.
- **3D brain renders** for Figure 4 (`data/anatomy_renders/*.png`). These are
  only re-rendered if you delete them, which needs `brainrender` and a display.

### Things that are deliberately *not* in git, and how to remake them

Most intermediate results are committed, so `figures` works from a clone
alone. The bulk below is not: either it is too large for GitHub, or it is a
per-session cache that a single script rebuilds. `.gitignore` lists them all
in one block. Nothing in the paper's main figures needs any of it -- these are
only wanted if you re-run a `derive` step or one of the dropped supplements.

| Not in git | Size | Remake it with |
|---|---|---|
| `data/waveforms/clark2025_waveforms.npz` | 1.4 GB | `pack_waveforms.py` (reads clark2025). Needed by the `isi_exclusion` derive step |
| `data/eye_frames/` | 1.5 GB | `cache_eye_frames.py`, which copies the eye videos from the lab share. Only needed for a Figure S17 trial that is not already in `eye_frames_M*D*_trials.npz` |
| `data/manifold_supp/` | 580 MB | `manifold_supp_build.py` (Figure S17's manifold panels read it) |
| `data/population_state/attractor_cache/` | 350 MB | `attractor_cache.py` (~2 min); read by the attractor search, reliability and multibump steps |
| `data/population_state/attractor_recruit/` | 63 MB | `attractor_recruit_examples.py`, after the search; for the dropped population-search supplement only |
| `data/lfp/lfp_power_by_anchoring.csv` | 218 MB | `lfp_batch.py` (needs clark2025_lfp). Over GitHub's 100 MB file limit. The summary tables the figures read *are* committed |
| `google_doc/` | 74 MB | `build_docx.py` (10 s) |
| `data/_PREVRBC_20261031/`, `data/_atlas_backup_20261026/`, `*_PREISI/`, `stale_*_PREISI/` | 740 MB | nothing: these are snapshots of `data/` taken before the curation and anatomy changes, kept locally for comparison only |

---

## 2. Point the scripts at your data (once)

The scripts were written on Harry's laptop and contain his paths. This command
replaces them with yours, everywhere they occur:

```bash
cd scripts/figures/AnchorDynamics2026
python rebuild_paper.py configure \
    --clark2025 /path/to/clark2025 \
    --lfp       /path/to/clark2025_lfp \
    --atlas     ~/.brainglobe/allen_mouse_10um_v1.2
```

Add `--dry-run` first if you want to see what it will change. It edits text
inside the scripts' string literals only, and records what it did in
`rebuild_paths.json`, so you can run it again later if the data moves.

## 3. Check

```bash
python rebuild_paper.py check
```

Every line should say `ok`. If you don't have clark2025_lfp yet, use
`check --skip-lfp`. A missing Arial font is only a note: the figures still
build, in a fallback font. To fix it on Linux, install the Microsoft core
fonts (`ttf-mscorefonts-installer`) and delete `~/.cache/matplotlib`.

## 4. Rebuild the figures and the paper

```bash
python rebuild_paper.py figures
```

This redraws every figure from the intermediate results in the repository and
then compiles `AnchorDynamics2026_Cell_mockup.pdf`. On Harry's laptop this took
about 6 minutes (without the two LFP figures), all 25 other steps succeeding.
Without clark2025_lfp, add `--skip-lfp`; Figures 9 and S12 then keep whatever
version is already in the folder.

Each step writes a log to `rebuild_logs/<step>.log`. A failed step doesn't
stop the rest, and the summary at the end lists anything that failed. To
re-run one step: `python rebuild_paper.py figures --only fig3_4`. To resume
from a step: `--from fig7`.

## 5. (Optional) Recompute the intermediate results from the raw data

```bash
python rebuild_paper.py derive          # then: python rebuild_paper.py figures
```

Step 4 uses the intermediate results already committed to the repository:
anchoring labels, the unit table, LFP spectra and so on. `derive` recomputes
all of them from clark2025 and clark2025_lfp, in order, so you can confirm the
committed versions follow from the data. It overwrites them in place, so
commit or copy `data/` first if you want to compare (`git diff --stat data/`
afterwards shows what changed). It is slow -- expect many hours, most of it the
LFP steps (`--skip-lfp` leaves those out) and `monosyn` -- and, unlike
`figures`, it has not yet been run end to end in one go: each script ran
individually while the paper was written, but if a step fails here its log
will say why, and Harry would like to know. Steps can be resumed with `--from`.

`relabel` reads `rule_shanks.csv` from the last `anatomy_rules` run, which takes
each shank's most dorsal curated unit as its brain entry, so a change to which
units are curated (the dorsal-island rule in `apply_isi_exclusion.py`) needs
`apply_isi_exclusion.py`, then `anatomy_rules.py` (cached spectra; no drive
needed), then `apply_rule_anatomy.py` and `apply_isi_exclusion.py` again,
before the derive.

Several steps resume from what they find on disk rather than recomputing, so a
derive after a change to the cells or labels (as after the ISI exclusion of
2026-10-30) must clear them first or it will quietly keep old results:
`data/population_state/labels/M*D*.npz` (`labels` reuses any session file it finds,
so a changed cell set needs these deleted -- found 2026-10-31, when it finished in 24 s),
`data/population_state/trial_map_stats.csv` (`trial_maps`),
`data/population_state/of_speed_score.csv` (`of_speed`, which skips every session
already in it -- found 2026-11-10, when it finished in 1 s),
`data/population_state/neighbour_test/` (`neighbour_test`, which keeps each
session's cell list), `data/monosyn/connections2_w*.csv` and `cells2_w*.csv`
(`monosyn`, now `run_monosyn.sh`, which runs the batch on every labelled
session), and the figure-stage caches `data/monosyn/pairs_by_identity.csv`,
`subnetwork_*.csv`, `connection_direction.csv`,
`data/population_state/identity_agreement.csv`, `s15_*`,
`vis_anchored_*.npz` and `data/eye_anchoring/eye_position_profiles_gated.npz`.
`labels` deletes old `anchoring_*_w*.csv` worker files itself, since the merge
takes every one it finds. `theta_freq` writes the per-trial state that
`speed_tuning` and `residual_speed` read, so the three run together, and all
three need the LFP drive. If the drive drops out mid-run, the LFP scripts skip
the sessions they cannot open without failing: check the session count in each
LFP log against the previous run.

---

## 6. Rebuild only the text

The four builders read markdown and the figure PDFs sitting next to them.
They never redraw a figure and never touch clark2025, so this part of the
rebuild works in a fresh clone with no dataset and no atlas:

```bash
python build_cell_pdf.py        # -> AnchorDynamics2026_Cell_mockup.pdf (+ _small)
python build_nature_pdf.py      # -> AnchorDynamics2026_Nature_mockup.pdf (+ _small)
python build_extra_pdf.py       # -> AnchorDynamics2026_Extra_analyses.pdf
python build_docx.py            # -> google_doc/AnchorDynamics2026_manuscript.docx
```

A couple of minutes each, most of it `shrink_pdf.py` resampling the images.
Always run `build_nature_pdf.py` after `build_cell_pdf.py`: it imports the
text parser, captions and references from it, so the two stay in step only if
both are rebuilt. `build_nature_pdf.py` needs `pip install pyphen` and the
faces that `extract_nature_fonts.py` pulls out once into `fonts/`.

### Where each piece of the text lives

| Piece | File | Notes |
|---|---|---|
| Title, Summary, Introduction, Results, Discussion | `MANUSCRIPT.md` | the Summary is the first body paragraph, under its own heading; the editorial matter at the end (AI declaration, editorial notes) is parsed out and placed separately |
| STAR Methods | `PAPER.md`, its `## Methods` section | `MANUSCRIPT.md` deliberately carries no Methods. `PAPER.md` is also the working document -- everything above `## Methods` (title options, the argument point by point, figure drafts) is not in the PDF |
| Extra analyses PDF text | `EXTRA.md` | same parser, same layout code |
| Figure captions | inside `build_cell_pdf.py` | one Python string per figure, next to the figure's filename; this is also the authoritative list of which file is which figure |
| References | the `REFS` dict in `build_cell_pdf.py` | cites are author-year in the markdown and resolved against `REFS`; **an unresolved key fails the build**, so add the entry before you add the citation. The Nature builder numbers the same entries in citation order |

So: a number or a sentence changes in `MANUSCRIPT.md` or in `PAPER.md`'s
Methods; a caption changes in `build_cell_pdf.py`; then rebuild both PDFs.
`PAPER.md` names the script behind each analysis in parentheses, and the
builder strips those filenames out of the Methods it sets, so you can keep
them there.

---

## Which script makes which figure

| Figure | Script |
|---|---|
| 1 | `figure1_schematics.ipynb` (panels), then `fig1_composite.py` |
| 2 | `fig1_supp_neighbour_bars.py` (reads `data/population_state/neighbour_test.csv`, made by `run_neighbour_labels.sh` → `build_neighbour_labels.py`, `neighbour_test.py`: ~15 min on 7 workers, resumable) |
| 3 | `fig2_single_units.py` |
| 4, S7 and S8 | `fig3_anatomy.py` (one script, three figures; S8 reads `data/population_state/fig4_factors.json`, from `fig4_factors.py`) |
| 5 | `fig4_monosynaptic.py` |
| 6 | `fig5_pupil_arousal.py` |
| 7 | `fig7_supp_vis_examples.py` |
| 8 | `fig7_theta_gamma.py` |
| 9 | `fig8_supp_attractor_more.py --sim` (A: M29 D25; B: the simulated module in `data/population_state/attractor_sim/sim.npz`, built by `attractor_sim_session.py` from `can_gate_model_2d.py` (`behave=True`, noise from `attractor_noise_calibration.py`), ~16 min; C-F: attractor numbers from `attractor_coherence.py`, bump speed over 2.5 s from `attractor_speed_cycle.py`; trials from `attractor_trial_types.py` (needs the VR files); open-field maps cached in `data/population_state/attractor_of1_M*D*.npz`) |
| 10 | `fig8_model.py` |
| S1 | `fig1_supp_classifier.py` (simulates its own grid cells; caches in `data/classifier_sim/`, ~20 min uncached) |
| S2 | `fig1_supp_population_state.py` (reads `data/classifier_sim/population_state_sim.npz`, made by `population_state_sim.py` from S1's cells, ~17 min; and `neighbour_test_pc1.csv`) |
| S3–S4 | `fig1_supp_rasters.py`, `fig1_supp_regions.py` |
| S5 | `fig1_supp_neighbour_test.py` (the same data as Figure 2) |
| S6 | `fig2_supp_classifier_steps.py`, and `fig2_supp_nonanchored.py`, stacked by `fig2_supp_nonanchored_rate.py` |
| S9 | `region_state_dimensionality.py` |
| S10–S13 | `fig7_supp_band_examples.py`, `fig7_supp_comodulogram.py`, `fig7_supp_gamma_locking.py`, `speed_tuning_fig.py` (S13 reads `speed_state_glm_cells.csv`, from `speed_state_glm.py`, and the drive for its example cell) |
| out of the paper | `fig8_supp_attractor_more.py` without `--sim` (the two-session version with reliability and one-bump panels, S14 until 2026-11-02; its analyses, `attractor_reliability.py` and `attractor_multibump.py`, are reported in the Methods as not shown) |
| S14–S16 | `fig_supp_lfp_spectra.py`, `fig_supp_histology_example.py`, `fig7_supp_more_examples.py` |
| S17 | `fig6_supp_pupil_examples.py` (eye-video stills cached in `eye_frames_M*D*_trials.npz`; a trial not cached there is read from the local copies `cache_eye_frames.py` took of the lab share, in `data/eye_frames/` -- not in git) |
| S18 | `fig5_supp_subnetwork.py` (caches `data/monosyn/subnetwork_{cells,coherence}.csv`, `connection_direction.csv`) |
| S19 | `fig5_supp_narrow_continuum.py` (narrow-spiking cells as a continuum of firing rate; caches `data/monosyn/pairs_by_cell.csv`) |
| out of the paper (tried as S15 on 2026-11-02, dropped) | `fig_supp_attractor_population.py` (reads `attractor_search_{sessions,cells,curves,eval,drift}{,_null}.csv` from `attractor_population_search.py`, `attractor_search_eval.py` and `attractor_search_drift.py`, each run twice, the second time with `NULL=1`; and `attractor_recruit_speed.csv` plus `attractor_recruit/` from `attractor_recruit_examples.py`, run after the search; all read `data/population_state/attractor_cache/`, from `attractor_cache.py`) |
| Extra E1 | `fig1_supp_of_decomp.py` (reads `data/of_xgb/of_xgb_cells.csv`, made by `run_of_xgb.sh` → `of_xgb_decomp.py`: ~5 h on 8 workers, resumable; and `data/of_xgb/of_xgb_examples.npz`, made by `build_of_xgb_examples.py`, ~2 min) |
| Extra E2–E4 | `fig3_supp_laminar.py`, `fig6_region_examples.py`, `fig6_supp_region_examples.py` |
| Extra analyses PDF | `build_extra_pdf.py` (text from `EXTRA.md`, captions inside the script) |
| not in the paper | `fig8_supp_can_model.py` (the gated attractor as a ring and a 2-D sheet; removed 2026-10-31, still built by `figures`) |
| not yet numbered: the disengagement supplement to Figure 6 (2026-11-20) | `pupil_disengagement.py` (~4 min, reads the VR files; writes `data/eye_anchoring/pupil_disengagement{.csv,_profiles.npz}`), then `fig6_supp_disengagement.py` → `fig6_supp_disengagement.pdf`. Built by `figures` but **not inserted into `build_cell_pdf.py`**, because inserting it renumbers S18 and S19 |
| not in the paper: reviewer questions answered off-figure (2026-11-17) | `pupil_stationary.py` (the pupil measure on stationary and black-box epochs, position-matched; writes `data/eye_anchoring/pupil_stationary.csv`, ~20 s) and `pupil_decode_state.py` (can the state be read from pupil alone: single-trial AUC, state recovery, change detection, each against 500 phase-randomised surrogates; writes `data/eye_anchoring/pupil_decode_state.csv`, ~20 min). Both import the paper's pupil measure by exec'ing two cells of `figure5_pupil_arousal.ipynb`, so they cannot drift from it |
| not in the paper | the mediolateral question is answered from `data/lfp/lfp_power_by_anchoring.csv`, which already carries `ml`, `ml_side`, `layer` and `dv` per channel group (`lfp_batch.py`, `mec_groups()`, split at `ML_BOUND = 3400` µm) -- no extra script |
| paper PDF | `build_cell_pdf.py` (text from `MANUSCRIPT.md` and `PAPER.md`, captions inside the script) |
| Nature-style PDF | `build_nature_pdf.py` (same text, figures and captions, imported from `build_cell_pdf.py`, set in Nature's Article layout; run `extract_nature_fonts.py` once for the Charter and Helvetica Neue faces, and `pip install pyphen` for hyphenation) |
| PDF size | `shrink_pdf.py`, called by both builders: each page's content streams merged into one (renders identically; a third of the file was their overhead), images resampled losslessly to 300 dpi (250 or 200 if still over 25 MB) in place, plus a `*_small.pdf` at 150 dpi; vector content untouched. Adds about 2 min per PDF |
| Google Doc | `build_docx.py` (10 s): the Cell mock-up's content as `google_doc/AnchorDynamics2026_manuscript.docx`, one column, figures as 200 dpi PNGs placed after their first citation (30 MB, under Google's 50 MB conversion limit), plus `google_doc/figures/` with each figure as PNG and vector PDF. Upload to Drive and open with Google Docs; edits made there do not flow back into MANUSCRIPT.md |

The file names lag the figure numbers because the figures were renumbered
during writing (most recently on 2026-11-02 (final): the simulated-module attractor figure became main Figure 9, before the model (now Figure 10), the two-session S14 was dropped, and S16-S21 closed up to S14-S19; before that, on 2026-11-02 (latest): a simulated gated-attractor session (attractor_sim_session.py) drawn as Figure S14 became S15, and S15-S20 became S16-S21; before that, on 2026-11-02 (later): the attractor supplement was rebuilt around two sessions (M29 D25, M25 D25) with the old S14's session-level panels and the one-bump-or-two test merged into it as S14, and S16-S21 closed up to S15-S20; before that, on 2026-11-02: more attractor sessions and the one-bump-or-two test became S15, after S14, and S15-S20 became S16-S21; before that, on 2026-11-01 (later): S5, S8, S11 and S12 moved to the Extra analyses PDF as E1-E4 and the rest closed up, S6-S24 becoming S5-S20; before that, on 2026-11-01: Figures 4 and 5 merged into Figure 4, its factor panel became S10, and old S9-S21 became S11-S23 after the shifts; before that, on 2026-10-31: old S21 became Figure 2, old
Figures 2-9 became 3-10, old S20 became S4, and the gated-attractor
supplement was removed; later that day the open-field decomposition left S3
for its own figure, and then the population-state comparison became S2; each
insertion moved the supplements after it up by one). Comments inside the scripts may still use older numbers. Each script's opening docstring explains what it computes and
why.

## If something goes wrong

- **A path error naming `/Users/harryclark/...`**: a path the `configure` step
  doesn't know about. Search for it (`grep -rn "/Users/harryclark" .`) and tell
  Harry. The ones that are expected to remain are only used when a cached
  input is missing (eye videos, old exploratory notebooks).
- **`check` reports clark2025 with 0 sessions**: the path should be the folder
  that contains `M20/`, `M21/`, ...
- **Figure 4 tries to open a 3D window**: a render PNG is missing from
  `data/anatomy_renders/`; restore it from git rather than re-rendering.
- **Numbers in a rebuilt figure differ from the PDF after `derive`**: most
  shuffles are seeded, but not all, so small differences in p-values are
  possible. Anything that changes a conclusion is worth telling Harry about.
- **A figure PDF the builder cannot find**: `*.pdf` is in `.gitignore`, so
  only the figures the builders actually read are committed (force-added). An
  exploratory PDF that a script writes alongside them is not in git, and does
  not need to be. Draw it with the figure step that makes it.
- **The reference PDFs**: the figure PDFs the paper is built from and the
  compiled `AnchorDynamics2026_Cell_mockup.pdf`, `..._Nature_mockup.pdf` and
  `..._Extra_analyses.pdf` are committed, so you can compare your rebuild
  against them.

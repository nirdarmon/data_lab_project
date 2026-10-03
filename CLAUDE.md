# CLAUDE.md - Concrete Crack Segmentation

Guidance for Claude Code in this repo. Read this first, then `.claude/rules/`
(methodology > style > ASCII). All prose here is plain ASCII per the global rule.

**Paths in this repo are always relative to the repository root.** Never hardcode an
absolute path in the notebook, a script, or documentation.

## What this is

Undergraduate Data Science Workshop project by a team of three, submitted as a
**rendered Jupyter notebook**. Two stages: the proposal (current work) and the final
report about a month later. Presented later as a ~20 minute video, so everything must
be explainable out loud.

Source of truth for requirements: `project_guideline.md` (Hebrew, supplied by the
course). It dictates the chapter list, demands ~50% prose / 50% code, and awards
credit for analysing failures rather than only showing successes.

## The problem - one task only

```
RGB image patch  ->  segmentation model  ->  binary mask (0 = background, 1 = crack)
```

Semantic segmentation. A crack-free patch has an empty target mask. There is **no**
classification model and no derived crack / no-crack label - this was ruled out
explicitly. Mask statistics are descriptive only, never model inputs.

Research question: can we automatically localize cracks in concrete at pixel level?

## Repository layout

```
proposal.ipynb                     Stage 1, submitted and APPROVED - frozen, do not edit
proposal.html                      rendered export of the proposal
final.ipynb                        THE CURRENT DELIVERABLE - the final report (Stage 2),
                                   rendered on the full train+val data
teacher_review.md                  supervisor feedback (Hebrew + English action items).
                                   The final report fixes ONLY what it lists.
examples/                          two previous students' notebooks, size/style
                                   reference only. Gitignored - not ours to share.
CLAUDE.md  README.md  requirements.txt
project_guideline.md               course requirements (Hebrew) - do not edit
data/
  concrete_crack_segmentation/     rgb/ and BW/, extracted from the .rar by Claude
  concreteCrackSegmentationDataset.rar   the data source, complete (see below)
  jwsn7tfbrp-1.zip                  Mendeley download - holds the TRUNCATED rar
outputs/                           generated artifacts, untracked except split.csv
docs/CODE_GUIDE.md                 study/presentation guide
.claude/rules/                     working rules
.claude/tools/check_ascii.py       ASCII linter
```

### Sizing the proposal - measured, do not re-litigate

The proposal must not read like a finished project. Measured against the two reference
notebooks in `examples/` (both are *completed final projects*):

| Notebook | cells | code cells | code lines | md words |
|---|---|---|---|---|
| Prostate Cancer (final) | 165 | 85 | 1071 | 5,309 |
| Data Sciense Project (final) | 519 | 297 | 2037 | 11,482 |
| extended draft (deleted) | 109 | 45 | 1133 | 8,575 |
| `proposal.ipynb` (as first written) | 47 | 18 | 603 | 2,355 |
| `proposal.ipynb` (as submitted) | 68 | 26 | 2,653 | 5,411 |
| `final.ipynb` (2026-09-28 render) | 68 | 26 | 2,807 | 5,621 |

Code lines are inflated by the one-argument-per-line formatting; the submitted
proposal and the final report are close in size because the final report so far
mostly re-ran the proposal on full data. The size cap applied to the proposal only.

The extended draft matched a finished project by volume, which is why it was replaced.
Keep `proposal.ipynb` at roughly half a final project. Markdown cells stay **3-6
lines**; the reference notebooks average 2-4 lines per markdown cell.

### Scope discipline - the goal is NOT to cover every method the guideline lists

Deliberately cut, and they stay cut unless the supervisor asks:

| Cut | Kept instead |
|---|---|
| UMAP | PCA + t-SNE |
| Agglomerative, DBSCAN | KMeans with silhouette |
| RandomForest | LogisticRegression, whose standardised coefficients ARE the explainability chapter |
| GroupKFold CV of the pixel model | the group-based train/val/test split itself |
| phash vs dhash bake-off | phash only; the dhash failure is stated in prose |
| 2-D black-hat tuning grid | fixed kernel 101 / threshold 160, justified in prose |
| per-cluster stratified evaluation | listed as final-project work |
| 23-feature filter bank | 7 features across 2 scales |

**Do not shrink the U-Net.** It stays four levels / 483k parameters. Depth is the
receptive field, and "context is what the deep model contributes" is the project's
central claim - cutting levels would undercut the argument and invalidate the verified
0.867 (0.882 on full data). Model capacity is not write-up complexity. The pretrained
ResNet34 U-Net (supervisor item 3) is added **next to** it as a comparison, not as a
replacement.

## Dataset - verified facts, do not re-derive

Ozgenel, "Concrete Crack Segmentation Dataset", Mendeley V1, DOI
`10.17632/jwsn7tfbrp.1`, CC BY 4.0. Lives at `data/concrete_crack_segmentation/`
with `rgb/` and `BW/` subfolders. The notebook locates it by globbing for a directory
containing both, so it does not depend on that exact name.

- 458 images, 458 masks, **stems match exactly**, zero orphans.
- `rgb/` mixes 257 `.jpg` and 201 `.JPG`; `BW/` is all `.jpg`. Pair by stem,
  case-insensitive on the extension. IDs are non-contiguous, running past 600.
- **EXIF orientation**: 257 absent, 179 = 6, 15 = 3, 7 = 1. Masks carry no rotation.
  Dimensions match 279/458 before `exif_transpose`, **458/458 after**. The 15
  Orientation=3 files match dimensions either way while being upside-down - shape
  equality is not proof of alignment. A content-based check (mean darkness inside the
  mask minus outside) scores 68.1 aligned vs 4.9 for 180-rotated, correct on 458/458;
  on those 15 it is 70.5 with the fix vs 6.0 without.
- Resolutions after correction: 259 + 179 at 4032x3024 (both orientations),
  13 + 7 at 3264x2448. Do not hardcode.
- Masks open in RGB mode; convert to `L`. **Crack is white (255).**
- **1.104% of mask pixels are exactly 255, 0.519% are 1-254** (JPEG ringing) - one
  ambiguous pixel per two crack pixels, so the binarization threshold matters.
- Crack ratio per image: mean 1.36%, median 1.17%, range 0.39-8.67%.
  **No image has an empty mask** - every source photo contains a crack, so empty
  patches arise only from cropping.
- Decode cost: EXIF only 0.0003 s/img, mask at 1/8 via `draft()` 0.008, full mask
  0.063, full RGB 0.106. Decoding is never the bottleneck; per-patch features and
  training are.

**How the data gets onto disk.** The team keeps the dataset as
`data/concreteCrackSegmentationDataset.rar`. **Claude extracts it** into
`data/concrete_crack_segmentation/{rgb,BW}` whenever those folders are missing or
empty - check the counts before any notebook run and extract without asking. The
whole `data/` folder is gitignored (`.gitignore` line 2), so the archive and the
extracted images never reach git - no extra ignore rule is needed. Do not re-download.
The archive root holds `BW/` and `rgb/` directly, so extract with (`-aos` skips files
already present):

```bash
"/c/Program Files/7-Zip/7z.exe" x data/concreteCrackSegmentationDataset.rar -odata/concrete_crack_segmentation -aos
```

Verify by counting files: `rgb/` and `BW/` must each hold 458.

**The archive is complete (replaced 2026-09-28).** `7z t` passes with 916 files
(458 rgb + 458 BW); after extraction every file decodes, 458/458 pairs, 0 orphans,
content alignment 458/458 at mean 68.1, EXIF counts as listed above.

**Beware Mendeley's own download.** `data/jwsn7tfbrp-1.zip` (the Mendeley "Download
All") wraps a `.rar` of exactly 714,931,556 bytes - the same truncated file as before:
"Unexpected end of archive", only 446 rgb (595, 596, 600, 602-610 absent) and
`593.JPG` cut short. Never replace the good 745,914,150-byte `.rar` with it.

The 40,000-image classification set commonly used for this topic was cropped from
these same photos but ships labels without masks, which is why we cut our own aligned
patches.

## Environment

conda env `kaggle`, Python 3.14.6, torch 2.13.0+cu126, torchvision 0.28.0+cu126
(source of the pretrained ResNet34), CUDA working on an RTX 3060
Laptop (6 GB). Also: opencv 5.0.0, scikit-image 0.26.0, scikit-learn 1.9.0,
pandas 3.0.5, numpy 2.4.6, pyarrow, imagehash, umap-learn 0.5.12, shap 0.52.0,
plotly 6.9.0. See `requirements.txt`.

```bash
conda activate kaggle     # always activate first, then call plain `python`
```

Windows + PowerShell. The bare `python` on PATH outside the env is a different 3.14
install without torch. The IDE's package warnings on `requirements.txt` come from it
inspecting that other interpreter, and can be ignored.

## Established results (proposal subset unless noted)

Split: 320 / 69 / 69 images, crack ratio 1.39 / 1.26 / 1.34%, frozen in
`outputs/split.csv`. Patch subset: 60 images -> 9,672 patches at 256x256,
83.4% completely empty, ~16% contain crack, consistent across splits.

Near-duplicates: phash at Hamming <= 8 gives 450 groups, largest 2, only 8
multi-image groups covering 16 images - duplication is minimal. dhash was rejected:
at distance <= 12 it collapsed 296 of 458 images into one component.

Validation micro Dice / IoU:

| Method | Dice | IoU |
|---|---|---|
| A0 predict all background | 0.000 | 0.000 |
| A1 Otsu on inverted gray | 0.046 | 0.024 |
| A2 CLAHE + blackhat + per-patch Otsu | 0.036 | 0.018 |
| A3 CLAHE + blackhat k=101 + global thr=160 (tuned on train) | 0.412 | 0.259 |
| B1 per-pixel LogisticRegression on filter bank | 0.572 | 0.400 |
| B2 per-pixel RandomForest | 0.540 | 0.370 |
| C U-Net smoke-train (15 epochs, 482,737 params) | **0.867** | **0.766** |

U-Net: base width 16, BCE+Dice, Adam 1e-3 with cosine annealing, AMP, batch 16,
2,216 balanced train patches, evaluated on the complete 1,485-patch val grid.
15.2 s/epoch on the 3060; best epoch 10; best threshold 0.55 (precision 0.911,
recall 0.827). **Without the cosine schedule the run was unstable** - val Dice hit
0.850 at epoch 8 then collapsed to 0.22 by epoch 13 while train loss kept falling.
Keep the schedule and the best-checkpoint selection.

**Both the epoch and the threshold were selected on validation, then validation Dice
was reported, so 0.867 is optimistically biased.** The test split is untouched and
reserved for the final report. Say this in the notebook.

Two more traps to keep stating in the notebook:
- **A0 scores 98.59% pixel accuracy and 0.838 macro Dice while predicting nothing.**
- **The GroupKFold pixel CV score (~0.885) is on a balanced pixel sample (24.4%
  crack), not a segmentation score.** It must never be compared to the 0.572 above.

A1's failure is explained, not hand-waved: Otsu must split every patch, so on empty
patches it labels 44.1% of pixels as crack versus 16.0% on cracked ones.

RandomForest is 8x slower than LogReg at inference (308 s vs 39 s for validation).

### Full-data results (`final.ipynb`, rendered 2026-09-28)

All 389 train+val images -> 63,330 patches (83.6% empty). Val grid: 69 images,
11,271 patches, 83.5% empty. U-Net trains on 16,630 balanced patches (8,315 crack +
8,315 empty), 20 epochs at ~145 s/epoch, best epoch 18, threshold 0.50 (the Dice curve
is flat within ~0.02 over 0.1-0.9). Logistic Regression trains on the same 8,315 + 8,315.

| Method (validation, micro) | Dice | IoU |
|---|---|---|
| Background | 0.000 | 0.000 |
| Otsu | 0.046 | 0.023 |
| Black-hat | 0.465 | 0.303 |
| Logistic Regression | 0.615 | 0.444 |
| U-Net (482,737 params) | 0.882 | 0.789 |
| ResNet34 U-Net (24,436,241 params, ImageNet encoder) | **0.898** | **0.815** |

U-Net precision 0.897 / recall 0.868. LogReg and U-Net have equal recall (86.8%); the
U-Net wins by ~10x fewer false-positive pixels. Background scores macro Dice 0.835 while
predicting nothing. Main U-Net failure: missed faint hairlines (worst five patches
Dice 0.00). On the 1,332 substantial-crack val patches, mean per-patch Dice is
Black-hat 0.511, LogReg 0.806, U-Net 0.896, ResNet34 U-Net 0.908. Test split still
untouched.

ResNet34 U-Net (supervisor item 3, run 2026-10-01): torchvision `IMAGENET1K_V1`
encoder (21.3M) + new decoder (3.2M); encoder frozen epochs 1-2 (BN in eval), then
fine-tuned at 1e-4 vs decoder 1e-3, same cosine schedule / batches / loss / epochs as
the plain U-Net via the shared `train_or_load_segmentation_network`. Frozen epochs
reach only val Dice 0.760 / 0.736 (below the plain U-Net); unfreezing jumps to 0.861.
Best epoch 19, threshold 0.60, precision 0.916 / recall 0.881; ahead of the plain U-Net
at every epoch from 7 on. vs plain U-Net: +136,592 TP pixels, -183,074 FP (-18.5%).
Same-patch figure: equal on wide breaks, still blind on faint hairlines (0.15 vs 0.01
on one, 0.00 both on the other). Cost: ~145 s frozen / 190-210 s fine-tuning epochs,
~64 min total vs ~48, weights 98 MB vs 2 MB. One seed per network - no variance
estimate. Best/worst examples and occlusion stay on the plain U-Net.

### Held-out test results (`final.ipynb` chapter 14, run 2026-10-03)

Test grid: 69 images, 11,100 patches, 83.2% without a crack (val 83.9%), 1.44% crack
pixels (val 1.34%). Scored once with all settings frozen from train/val.

| Method | Val Dice | Test Dice | Test IoU |
|---|---|---|---|
| Background | 0.000 | 0.000 | 0.000 (98.6% accuracy, macro Dice 0.827) |
| Otsu | 0.046 | 0.048 | 0.024 |
| Black-hat | 0.465 | 0.486 | 0.321 |
| Logistic Regression | 0.615 | 0.660 | 0.493 |
| U-Net | 0.882 | 0.870 | 0.770 |
| ResNet34 U-Net | 0.898 | **0.886** | **0.795** |

Both U-Nets drop exactly 0.012 (val selection of epoch + threshold); non-deep methods
rise (no val selection; more crack favours over-predictors - LogReg precision
0.476 -> 0.542). The pretrained gain is again 0.016 on test (+211,310 TP, -93,917 FP).

## Compute tiering

Cheap steps run on everything, expensive steps on a subset, and every table says
which. In `final.ipynb` one `PATCH_STAGE_SOURCE_IMAGE_COUNT` constant controls it
(`None` = all 389 train+val images, used for the rendered report; a small number gives
a quick smoke run). The proposal used `PROPOSAL_SUBSET_N` = 60.

| Tier | Scope | What |
|---|---|---|
| 0-1 | **all 458** | Pairing, EXIF scan, resolutions, per-image crack ratio and mask histogram |
| 2 | **all 458** | The train/val/test split |
| 3 | train+val (60 in the proposal) | Patch generation, metadata table, EDA, PCA/t-SNE/UMAP, clustering |
| 4 | train+val (60 in the proposal) | Classical baseline, per-pixel sklearn baseline, U-Net smoke-train |

## How the rubric is covered without a second task

| Guideline ch. | Covered by |
|---|---|
| 5 data structure | The patch metadata table - one row per patch |
| 6 cleaning, leakage | Stem pairing, EXIF fix, binarization, near-duplicate grouping, group split |
| 7 EDA, PCA/t-SNE/UMAP, clustering | Run on patch descriptors; clusters reused as **evaluation strata** (Dice per visual regime) |
| 8 simple + advanced models | Simple = classical thresholding and the per-pixel sklearn filter-bank model; advanced = U-Net. Both output masks. |
| 9 evaluation, confusion matrix | Train/val Dice curves, val vs held-out **test** (chapter 14), **pixel-level** confusion matrices. No cross-validation in `final.ipynb` - chapter 14 explains why (k retrains per U-Net; the group split already guards against a lucky split) |
| 11 explainability | Permutation importance / SHAP over filter-bank features; occlusion sensitivity for U-Net |

## Locked-in decisions (do not re-litigate)

- Segmentation only. No classification task.
- **All code lives in the notebook** (`final.ipynb` now). No `src/` package - chosen so the grader
  sees every line. Helpers go in cells near the top of their chapter.
- English-only markdown, technical, not high-level.
- Patches are **not** written to disk. `outputs/patch_index_<hash>.parquet` holds one
  row per patch (`image_id`, `split`, `y0`, `x0`, `size`, `crack_pixels`, `crack_ratio`,
  descriptors) and crops are taken lazily from the source images.
- **Caching in `final.ipynb`.** `build_cache_path()` (cell 5) names a cache file in
  `outputs/` by a hash of every parameter that shapes it. Six flags at the top of cell 5,
  each loading its cache only if a matching file exists (otherwise the step runs and
  writes it; `False` always recomputes and overwrites):
  `USE_CACHED_PATCH_TABLE` (`patch_index_<hash>.parquet`),
  `LOAD_TRAINED_LOGISTIC_REGRESSION` (`logistic_regression_<hash>.joblib`),
  `USE_CACHED_BASELINE_SCORES` (`val_scores_<method>_<hash>.json`, the val-grid scores
  and pixel counts of Background, Otsu, Black-hat and Logistic Regression),
  `LOAD_TRAINED_UNET` (`unet_<hash>.pt`, best weights + training history),
  `LOAD_TRAINED_PRETRAINED_UNET` (`unet_resnet34_<hash>.pt`, same contents), and
  `USE_CACHED_TEST_SCORES` (`test_scores_<hash>.json`, all six methods on the test grid,
  keyed on both checkpoints and thresholds). Code is not in
  the key - after editing a method's code, delete its file in `outputs/`.
- Data is never re-downloaded; a `resolve_data_root()` helper globs for the folder.
- Patch size 256x256, stride 256 for val/test grids.
- Notebook is committed **with outputs** - the submission must be rendered. Do not
  strip them. `outputs/split.csv` is force-added despite `.gitignore`.

## Phase status

1. Scaffolding, rules, CLAUDE.md - DONE.
2. Analysis code developed and verified as scripts - DONE (all stages incl. U-Net).
3. `proposal.ipynb` - DONE, rendered end to end, submitted with names, **approved**
   by the supervisor (`teacher_review.md`). Frozen from here on.
4. `docs/CODE_GUIDE.md` - written for the proposal; needs a pass for `final.ipynb`.
5. Final report `final.ipynb` (Stage 2) - IN PROGRESS. Done: full-data run with
   caching, text refreshed to the full-data numbers (commit `54c8617`); pretrained
   ResNet34 U-Net comparison (76 cells). Full renders are started by the team, not
   Claude - Claude smoke-tests first (6 images, 3 epochs, scratchpad caches).

### Still open - the four supervisor items (`teacher_review.md`)

| # | Item | Owner |
|---|---|---|
| 1 | Loss for extreme imbalance - DONE 2026-10-03 as prose only: KEEP BCE + soft Dice (no Tversky retrain). Cell 42 of `final.ipynb` answers his closing question, explains each term under 98.6% background, and why not Tversky (alpha = beta = 0.5 is Dice; the val threshold already trades precision for recall with flat Dice; Tversky left as future work for faint hairlines) | Nir (with Claude) |
| 2 | Fixed crack:background train patch ratio - DONE 2026-10-03 as prose only: the team KEEPS 50/50 (8,315 + 8,315; no 60/40 retrain). The proposal only said "a similar number", so the supervisor missed it; cells 20 and 44 of `final.ipynb` now state and justify it (flat threshold curves, 29 of 8,315 background patches hold a 1-49 px sliver, balanced set is 4.7% crack pixels) | Nir (with Claude) |
| 3 | Pretrained encoder: ResNet34 U-Net vs plain U-Net - DONE 2026-10-03 (chapter 11.4, prose written from the full run) | Nir (with Claude) |
| 4 | Post-processing (closing, small-component removal, skeleton) and its Dice/IoU effect | teammates |

Items 1 and 2 were closed in prose, so no retrain is planned: the current
checkpoints, validation and test outputs are final unless item 4 or a new decision
changes training.

Also open:
- Test evaluation DONE (chapter 14, 2026-10-03), prose written from the run. Re-run it,
  and refresh its prose, only if training ever changes.
  Chapters 15-17 are Tools / Limitations / Summary and conclusions.
- `docs/CODE_GUIDE.md` still describes the proposal.
- The builder scripts are gone; edit notebooks directly (`NotebookEdit` or a JSON
  script that keeps cell ids and trailing newlines).

### Gotchas that already cost a failed run

**1. Import order in the imports cell (cell 5 of `final.ipynb`) is load-bearing. Do not "tidy" it.**
scikit-image and scikit-learn link Intel OpenMP via MKL, and so does PyTorch. If
`torch` is imported first, a later `from skimage.feature import ...` **kills the
kernel**:

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

nbconvert reports this only as `DeadKernelError: Kernel died`, with no traceback and
no indication of which cell. Fix: import numpy/pandas, then skimage/sklearn/umap, then
cv2/matplotlib, and **torch last**. Verified: skimage-before-torch works;
skimage-after-torch aborts. The `KMP_DUPLICATE_LIB_OK=TRUE` override also works but is
officially unsafe, so we rely on ordering instead. All imports live in that cell; do not
add local imports to later cells.

**2. `nbformat` needs each `source` line to keep its trailing newline.**
Splitting on `"\n"` without re-adding them concatenates the whole cell onto one line
and every cell fails with `SyntaxError`.

**3. Cells need an `id` field**, or nbformat emits `MissingIDFieldWarning`.

Before launching a ~20 minute execution, cheaply pre-flight the notebook:
`compile()` every code cell, check every cell has an `id`, and run
`nbformat.validate`. To locate a crash, flatten the notebook to a script with a
`print(">>> CELL n")` marker before each cell and run that instead - it gives the
exact failing cell, which nbconvert does not.

## Working style

- The team prefers momentum and batched progress; confirm genuinely
  decision-changing choices before building.
- They want to *understand* the code, not just receive it - hence `docs/CODE_GUIDE.md`.
- Git repo, delivered as a rendered notebook or HTML. The working copy currently sits
  in iCloudDrive, so large writes may lag while syncing.

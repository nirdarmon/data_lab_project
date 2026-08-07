# Code Guide - Concrete Crack Segmentation

A study companion to `proposal.ipynb`. The notebook is written for a grader; this
document is written for **us**, so that every one of the three of us can explain any
part of the project without having authored it.

Read this alongside the notebook. It covers, per chapter: what the code actually does,
the concept behind it, and what to say when presenting.

**Contents**

1. [How to run it](#0-how-to-run-it)
2. [Concept primers](#1-concept-primers)
3. [Chapter-by-chapter walkthrough](#2-chapter-by-chapter-walkthrough)
4. [Presentation plan](#3-presentation-plan)
5. [Questions we should expect](#4-questions-we-should-expect)

---

## 0. How to run it

```bash
conda activate kaggle
jupyter lab
# open proposal.ipynb -> Kernel > Restart and Run All
```

Roughly 20 minutes end to end, most of it in the U-Net section. Requires the dataset
under `data/` (see `README.md`) and a CUDA GPU for the training section - it falls
back to CPU but will be very slow.

Everything is seeded from `SEED = 42`. All paths are relative to the repo root.

---

## 1. Concept primers

Read this section once. Everything in the notebook builds on these ideas.

### 1.1 Semantic segmentation

Classification assigns one label to a whole image. **Semantic segmentation assigns a
label to every pixel.** Our output is the same height and width as the input, with one
number per pixel: the probability that the pixel is crack. Threshold it and you get a
binary mask.

Think of it as classification run 65,536 times per patch, except the predictions are
not independent - the model uses context.

### 1.2 Dice and IoU

Both measure **overlap** between the predicted mask and the true mask. With `TP` true
positives, `FP` false positives and `FN` false negatives, all counted in pixels:

```
Dice = 2*TP / (2*TP + FP + FN)
IoU  =   TP / (TP + FP + FN)
```

- Dice is the same thing as the F1 score, applied to pixels.
- IoU is "intersection over union": the shared area divided by the total area covered
  by either mask.
- **IoU is always <= Dice.** They rank models identically, so never compare a Dice
  number to an IoU number and conclude anything. Dice 0.87 corresponds to IoU 0.77.

Why not accuracy? Because 98.9% of pixels are background. Predicting "no crack
anywhere" gets 98.6% accuracy and Dice 0.000. We show exactly this in Chapter 12 - the
`A0` row exists purely to make the point.

### 1.3 Micro vs macro averaging

Two ways to summarise a metric over many patches:

- **Micro**: add up TP, FP, FN across all patches, then compute Dice once. Every
  *pixel* counts equally. A patch with a big crack influences the result more than one
  with a small crack.
- **Macro**: compute Dice per patch, then average. Every *patch* counts equally.

They disagree enormously here. 83.4% of our patches are empty, and by our convention
an empty prediction on an empty target scores a perfect 1.0. So macro Dice is
dominated by patches where doing nothing is correct - the do-nothing model scores
**0.838 macro** and **0.000 micro**.

**We lead with micro.** Whenever we quote a Dice number we say which one it is.

### 1.4 The empty-mask convention

Dice is `0/0` when a patch has no crack and the model predicts none. Undefined. Since
that is the *common* case here, we must choose:

> If `TP = FP = FN = 0`, score **1.0**. Predicting nothing on an empty target is
> correct.

Any convention is defensible; not stating one is not.

### 1.5 EXIF orientation - the bug we found

Cameras often store the photo in the sensor's native orientation and add a metadata
tag saying "rotate this by 90/180/270 when displaying". That tag is **EXIF
Orientation**.

`PIL.Image.open()` reads the pixels but **does not apply the tag**. The tool that
produced our masks did apply it and saved the mask already rotated. Result:

- 179 images tagged "rotate 90": stored 4032x3024, mask 3024x4032. Dimension mismatch
  - catchable.
- **15 images tagged "rotate 180": dimensions identical either way, content
  upside-down. Not catchable by any shape check.**

`ImageOps.exif_transpose(im)` applies the tag. We call it on every RGB load. To prove
alignment we use a content-based test: cracks are darker than concrete, so the mean
darkness *inside* the mask should exceed the mean darkness outside. Score 68.1 for
correct orientation, 4.9 for 180-rotated, correct on 458/458.

**This is the single best story in the project.** It is a real bug, found by us,
invisible to conventional checks, and fatal if missed.

### 1.6 Black-hat morphology

Two primitive operations on a grayscale image:

- **Erosion** shrinks bright regions; **dilation** grows them.
- **Closing** = dilate then erode. It fills in dark structures narrower than the
  structuring element, leaving the "background" as if the thin dark things were gone.

**Black-hat = closing - original.** What survives is exactly the dark structures
*thinner than the structuring element* - which is a far better description of a crack
than "dark pixel". Crucially it also cancels slow brightness gradients, so uneven
lighting stops mattering.

The structuring element size sets the maximum crack width detected. We swept it and
found performance plateaus around 101 pixels.

### 1.7 Otsu thresholding, and why it failed

Otsu picks the threshold that best separates a grayscale histogram into two groups. It
works beautifully when there really are two groups.

**It is obliged to return a split even when there is only one group.** On an empty
patch of uniform concrete, Otsu splits sensor noise - and labels 44.1% of pixels as
crack. Since 83.4% of patches are empty, these false positives destroy the pooled
precision.

The fix is not better morphology; it is a **global threshold learned on training
data**, which is allowed to answer "nothing here". That change alone took micro Dice
from 0.046 to 0.412.

### 1.8 The Hessian ridge detector

The Hessian is the matrix of second derivatives - it describes local curvature. Its
**eigenvalues** tell you the curvature along the two principal directions.

For a dark line on a light background: curvature is large across the line, near zero
along it. That signature is what a "ridge/valley detector" looks for, and it is the
standard tool for blood vessels, roads and cracks. We compute it at sigma in
{1, 2, 4, 8} so cracks of different widths are all detected.

### 1.9 U-Net

An encoder-decoder network shaped like a U.

- **Encoder** (left side): repeated convolution + max-pooling. Each pooling halves the
  resolution and doubles the channels. Pooling grows the **receptive field** - how much
  of the original image one output pixel can "see". Context is what lets the network
  distinguish a crack from a shadow edge.
- **Bottleneck**: lowest resolution, richest features.
- **Decoder** (right side): **transposed convolution** upsamples back toward full
  resolution. A transposed convolution is a learnable upsample - roughly, spread each
  input pixel out and convolve, rather than just repeating pixels.
- **Skip connections**: the output of each encoder level is concatenated onto the
  matching decoder level.

**Why skip connections are the whole point.** Pooling gives context but destroys
spatial precision. You cannot outline a 2-pixel-wide crack from a feature map that has
been downsampled 8x. Skip connections hand the decoder the un-pooled, full-detail
features, so the network gets context *and* precision. Without them a U-Net produces
blobby masks.

Ours is small: base width 16, four levels, 482,737 parameters.

### 1.10 Why BCE + Dice loss

- **Binary cross-entropy** scores each pixel independently. With 98.9% background, the
  gradient from the majority class dominates and the network converges to predicting
  nothing - which BCE considers a very good solution.
- **Dice loss** = `1 - Dice`, computed differentiably from probabilities. It is a
  *ratio*, so it is insensitive to how frequent the positive class is. It keeps
  pushing on the rare crack pixels.

We sum them: BCE for well-calibrated per-pixel probabilities, Dice for overlap
awareness. This is standard practice in medical and crack segmentation, and it is the
single most consequential choice for an imbalanced problem.

### 1.11 Group-based splitting and leakage

Ordinary random splitting is wrong when rows are related. Two patches from the same
photograph share lighting, texture and often the same crack; if one lands in train and
the other in validation, the model can score well by recognising the wall.

So the split is assigned to **source images before any patch is cut**, and actually to
near-duplicate *groups* of images. Every patch inherits its parent's split and carries
`image_id`, and the notebook asserts the three id sets are disjoint.

(scikit-learn's `GroupKFold` applies the same idea to cross-validation. We do not use
it in the proposal - cross-validating is final-project work - but it is the tool to
reach for there.)

### 1.12 Perceptual hashing

A perceptual hash maps an image to a short bit string so that visually similar images
get similar strings; **Hamming distance** counts differing bits.

We used it to find near-duplicate photographs (same wall, two shots), because those
would leak across splits.

The lesson worth presenting: **we validated the threshold instead of copying one.**
The common advice is "distance <= 10 means duplicate". With `dhash` that merged **296
of 458 images** into one group, because flat grey concrete has near-identical
neighbouring-pixel gradients regardless of which wall it is. `phash` was stable, so we
used phash at distance 8. The honest result: only 8 duplicate pairs exist. Small
correction, but we can prove it is small.

### 1.13 PCA and t-SNE

Both reduce many dimensions to two for plotting.

- **PCA** - linear, deterministic, preserves global variance. Fast, and the explained
  variance ratio is interpretable. Cannot unfold curved structure.
- **t-SNE** - non-linear, preserves *local* neighbourhoods. `perplexity` (we use 30) is
  roughly how many neighbours each point is fitted to; 30 is the standard middle of the
  useful 5-50 range. Distances *between* clusters in a t-SNE plot are not meaningful.

We show both because agreement between a linear and a non-linear view is evidence the
structure is real. (UMAP would be a third option; we dropped it to keep the proposal
lean and to avoid the dependency.)

### 1.14 Silhouette score

For each point: how much closer is it to its own cluster than to the nearest other
cluster? Averaged over all points, in [-1, 1]. Near 1 is well separated, near 0 means
clusters overlap, negative means points are in the wrong cluster.

We use it to choose k for KMeans instead of guessing.

### 1.15 AMP (mixed precision)

`torch.autocast` runs most operations in 16-bit instead of 32-bit. Faster, roughly
half the memory, which matters on a 6 GB laptop GPU. `GradScaler` multiplies the loss
before backward so small gradients do not underflow in 16-bit, then unscales before
the optimiser step.

---

## 2. Chapter-by-chapter walkthrough

### Ch. 1-3: Problem, motivation, data

No code. The arguments to be able to make:

- **Why segmentation and not classification.** A classifier says "there is a crack".
  On a structure already known to be cracked that is useless. Engineers need *where*
  (to revisit), *how big* (to grade severity) and *has it grown* (to predict failure).
  All three need a per-pixel answer.
- **Why this dataset.** The famous 40,000-image crack dataset was cropped from these
  same 458 photos but ships only labels, no masks. It cannot train a segmentation
  model. Working from the originals lets us cut aligned image+mask patches and control
  the split.

### Ch. 4: Setup

`find_dataset_root()` globs for any folder containing both `rgb/` and `BW/`, so no
path is hardcoded and the extracted folder can be called anything.

Three loading functions, and **all image access goes through them**:

| Function | Purpose |
|---|---|
| `load_rgb_image` | Applies `exif_transpose`. Non-negotiable - see 1.5. |
| `load_binary_mask` | Converts to grayscale and thresholds at 127. |
| `load_gray_downscaled` | 1/8-scale decode using JPEG `draft()`, ~8x faster, for whole-dataset passes. |

`draft()` is worth understanding: it decodes at reduced size *inside* the JPEG
decoder by discarding high-frequency DCT coefficients, rather than decoding fully and
resizing. 0.008 s/image vs 0.063 s.

### Ch. 5: Data structure

The conceptual core. Three levels:

| Level | Count | Role |
|---|---|---|
| Source image | 458 | The unit the **split** is assigned to |
| Patch | ~165/image | One **row** of the metadata table; what the model consumes |
| Pixel | 65,536/patch | What is **predicted** and scored |

The **patch metadata table** is our answer to "what does one row represent". We store
grid coordinates plus descriptors and crop lazily, rather than writing ~75,000 JPEGs.

Note carefully: the descriptor columns (`edge_density`, `glcm_contrast`, ...) are for
**EDA and error stratification only**. They are never model inputs. Be ready to say
this - it is an obvious question.

### Ch. 6: Cleaning - our strongest chapter

Five checks in order:

1. **Pairing** by filename stem, case-insensitively (`rgb/` mixes 257 `.jpg` with 201
   `.JPG`; `BW/` is all lowercase). Result: 458 pairs, no orphans.
2. **Dimensions** before and after `exif_transpose`: 279/458 -> 458/458.
3. **Alignment by content** - the check that catches the 15 invisible failures.
4. **Mask binarity**: 1.104% of pixels are pure white, but **0.519% are intermediate**
   greys from JPEG ringing. One ambiguous pixel per two crack pixels. We threshold at
   127 and say so, because 40 or 200 would visibly thicken or thin every crack.
5. **Crack coverage**: mean 1.36% per image. And **no image has an empty mask** -
   every photo contains a crack, so all our empty patches come from our own cropping.
   That is a real limitation (Ch. 15).

### Ch. 7: Split

Assigned to **source images before cropping**, and specifically to **near-duplicate
groups**, stratified by crack coverage into quartiles. 320/69/69, crack ratios
1.39/1.26/1.34%.

Two assertions print their result rather than passing silently - they are evidence,
not scaffolding.

The split is written to `outputs/split.csv` and **frozen**. The final report loads it
instead of recomputing, so the leakage argument does not have to be re-proven.

**After this chapter the test split is never touched again in the proposal.**

### Ch. 8: Patches

- **Crop, never resize.** Downscaling 4032x3024 to 256x256 reduces a 3-pixel crack to
  a fraction of a pixel and interpolates it away. This is the most important sentence
  in the chapter.
- Image and mask are cropped **in the same call with the same coordinates**, so
  misalignment is structurally impossible.
- Grid computed per image (resolutions and orientations vary). Remainder strips under
  256 px are dropped rather than padded - padding invents content.
- 60 images -> 9,672 patches, 83.4% empty, ~16% with crack, consistent across splits.
- **Sampling policy**: training rebalances (all cracked patches + equal empties);
  **validation and test keep the complete unsampled grid**. Rebalancing an evaluation
  set inflates every score.

The visual check - ground-truth contour drawn on the image patch - is what a human can
actually verify. It also reveals that masks are **thicker than the cracks**, which
caps achievable Dice.

### Ch. 9: EDA

Framed as: *what visually distinct kinds of patch exist, and will the model be equally
good at all of them?*

- **Correlation**: two redundancy blocks (brightness; local variation). Texture and
  edge descriptors correlate with crack coverage more than brightness does - so a
  crack is not merely "a dark region".
- **PCA and t-SNE**: the key observation is that crack-bearing patches are **not
  cleanly separated** in descriptor space. That is a useful *negative* result: if they
  were separable, a simple model on seven features would solve the problem and we would
  not need a U-Net.
- **Clustering**: KMeans, with k chosen by silhouette score rather than guessed. The
  clusters correspond to surface regimes (smooth, rough, shadowed), and crack patches
  appear across several of them - so they describe *conditions the model must work
  under*, independent of the label.

Reporting Dice **per cluster** is listed as final-project work. Be ready to say that,
because "clustering because the rubric asked for it" is the obvious criticism.

### Ch. 10-11: Models and protocol

Three approaches, all solving the same task so all directly comparable. See primers
1.6-1.10.

Protocol discipline to state out loud: thresholds and hyperparameters come from
train/validation only; validation and test use deterministic unsampled grids; the test
split is untouched in the proposal.

### Ch. 12: Results

The narrative:

| Model | micro Dice | What it adds |
|---|---|---|
| A0 do nothing | 0.000 | Exposes that accuracy is 98.6% and macro Dice 0.838 |
| A1 Otsu | ~0.05 | Cannot abstain - 44% positives on empty patches |
| A2 blackhat + global threshold | ~0.41 | **~9x from fixing the decision rule, not the features** |
| B per-pixel LogReg | ~0.57 | Learned decision on 7 named multiscale features |
| C U-Net | ~0.85 | Learned features **plus spatial context** |

Two things to be honest about, unprompted:

1. **The U-Net number is optimistically biased** - we chose both the epoch and the
   threshold on validation and then reported validation.
2. **Without the cosine LR schedule the training collapsed** (val Dice 0.85 at epoch 8
   -> 0.22 at epoch 13 while train loss kept falling). We report this because it is a
   real result about how fragile training is at this data scale.

The A1 -> A2 jump is the most quotable result: identical morphological idea, but a
global threshold that is *allowed to answer "nothing here"* replaces a per-patch Otsu
that is obliged to split every patch. The representation was never the bottleneck; the
decision rule was.

### Ch. 12: Explainability

The logistic regression was fitted on **standardised** features, so its coefficients
are directly comparable - the coefficients *are* the explanation, with no extra tooling.
This is exactly why we chose LogReg over a random forest here: interpretability comes
free, and a proposal does not need SHAP.

Expect a negative weight on smoothed intensity (darker pixels are more likely crack) and
positive weights on the ridge and gradient responses. The important part is that **more
than one scale contributes**: the model combines evidence about structures of different
widths, which is what you would design by hand for cracks of varying thickness.

This also explains why the U-Net wins: our filter bank is fixed, local and capped at
sigma = 4, while the U-Net learns its filters and sees far more context.

### Ch. 14-15: Limitations and plan

Know the four data limitations cold: single campus; **no crack-free source images**;
lossy masks; single annotator with no agreement estimate.

---

## 3. Presentation plan

~20 minutes, three presenters. Suggested split, each with a natural handover.

### Presenter 1 (~7 min) - problem and data quality

1. The problem and why **segmentation, not classification** (1 min).
2. Motivation: inspection cost, consistency, propagation tracking (1 min).
3. Dataset, and why we work from the 458 originals rather than the 40k crops (1 min).
4. **The EXIF bug** (3 min) - the centrepiece. Walk it: dimensions mismatch on 179,
   fixed by `exif_transpose`; then the 15 images where dimensions match but the image
   is upside-down; then the content-based alignment test, 68.1 vs 4.9 on 458/458.
5. Masks are lossy JPEG: one ambiguous pixel per two crack pixels (1 min).

*Handover: "so the data is clean - now, how do we make sure we do not cheat?"*

### Presenter 2 (~6 min) - leakage, split, EDA

1. Why patches from one image cannot straddle the split (1 min).
2. Near-duplicate detection, and **why we validated the hash threshold** - dhash
   merging 296 of 458 images is the memorable number (2 min).
3. The frozen stratified split and the assertions (1 min).
4. Patch generation: crop not resize, identical coordinates, 83.4% empty (1 min).
5. EDA: the descriptors do **not** separate cracked from clean patches, which is the
   argument for learned features (1 min).

*Handover: "hand-crafted features are not enough - so what happens when we model it?"*

### Presenter 3 (~7 min) - metrics, models, results

1. **Why accuracy is banned**: the do-nothing model scores 98.6% (1.5 min). Dice, IoU,
   micro vs macro, the empty-mask convention.
2. The three model families (1.5 min).
3. The results progression 0.41 -> 0.57 -> 0.87, and the insight that A1 to A3 was a
   **decision-rule fix, not a feature fix** (2 min).
4. Qualitative panel and failure modes (1 min).
5. Limitations and the final-project plan (1 min).

### Rehearsal notes

- Pre-run the notebook before recording; do not execute the U-Net live.
- Have `outputs/model_comparison.csv` open as a fallback if a figure misrenders.
- Whoever presents Ch. 12 must be able to answer "is 0.87 real?" - see below.

---

## 4. Questions we should expect

**"Why not just use the 40,000-image dataset?"**
It has labels but no masks. It cannot train a segmentation model. It was cropped from
these same 458 photos, so we are using the source rather than the derivative.

**"Is 0.87 Dice real?"**
It is optimistically biased and we say so in the notebook. We selected both the epoch
and the threshold on validation and then reported validation. It comes from 42
training and 9 validation images. The unbiased number will come from the untouched
test split in the final report, and we expect it to be lower.

**"Why is your macro Dice so different from micro?"**
83.4% of patches are empty, and by our stated convention an empty prediction on an
empty target scores 1.0. So macro is dominated by patches where doing nothing is
correct - the do-nothing model scores 0.838 macro. We lead with micro.

**"Why did you cluster? Is it not decoration?"**
The clusters describe visual regimes - conditions the model has to work under,
independent of the label. In the final project we report Dice per cluster, which turns
one average into a statement about *where* the model struggles.

**"You have no classification model - does the rubric not ask for classification?"**
The rubric asks for simple and advanced models. Ours are the classical baseline, the
per-pixel logistic regression and the U-Net, and all three solve the same segmentation
task, so they are directly comparable. Segmentation strictly contains the presence
question anyway.

**"Why so few methods? The guideline lists many more."**
The proposal is scoped deliberately. We show one classical baseline, one simple ML
baseline and one deep model, plus one dimensionality-reduction comparison and one
clustering method. Adding UMAP, DBSCAN, a random forest and cross-validation would have
made the proposal the size of a finished project without changing a single conclusion.
The extended exploration is kept in `drafts/proposal_extended.ipynb`, and the final
report expands from there.

**"How do you know your masks are aligned?"**
Two independent guarantees. Structurally, image and mask are cropped in the same call
with the same coordinates. Empirically, the darkness-based alignment score picks the
correct orientation on 458/458 with a wide margin.

**"Your model may just detect dark lines, not cracks."**
That is a genuine risk and we list it as an open question. The failure cases include
mortar joints and shadows. Larger context and more training data are the planned
mitigations.

**"Why 256x256?"**
Large enough to contain useful context, small enough that a batch fits in 6 GB. We
plan to ablate 256 vs 512 and a downscale-then-crop variant in the final project.

**"What if the annotations are wrong?"**
The masks are visibly thicker than the cracks, so some scored false positives at
boundaries are annotator imprecision. With a single annotator and no agreement
estimate, we cannot separate the two - which is why we call it a ceiling on achievable
Dice rather than model error.

# Concrete Crack Detection and Localization Using Deep Learning

Project for the Data Science Workshop. The deliverable is a rendered Jupyter
notebook; see `proposal.ipynb`.

## Problem

Semantic segmentation of cracks in concrete surfaces. One prediction task:

```
RGB image patch  ->  segmentation model  ->  binary mask (0 = background, 1 = crack)
```

A crack-free patch has an empty target mask. There is no separate crack / no-crack
classification model.

## Dataset

Ozgenel, C.F., "Concrete Crack Segmentation Dataset", Mendeley Data, V1 (2019).
DOI: [10.17632/jwsn7tfbrp.1](https://doi.org/10.17632/jwsn7tfbrp.1) - licensed CC BY 4.0.

458 high-resolution photographs of concrete surfaces with manually annotated binary
crack masks:

| Folder | Contents |
|---|---|
| `data/concrete_crack_segmentation/rgb/` | Source photographs (mixed `.jpg` / `.JPG`, mostly 4032x3024) |
| `data/concrete_crack_segmentation/BW/`  | Manually annotated masks, white = crack |

This is the source dataset from which the widely used 40,000-image crack
*classification* set was cropped. Because that derived set ships labels only and no
masks, we cut our own aligned image/mask patches from these originals instead.

The dataset is **not** committed. Download it from the DOI above and extract it under
`data/`; the notebook locates it automatically by searching for a directory
containing both `rgb/` and `BW/`.

## Repository layout

```
proposal.ipynb                  the proposal - submitted and approved
proposal.html                   rendered export of the proposal
final.ipynb                     the final report; starts as a copy of the proposal
requirements.txt
project_guideline.md            course requirements (Hebrew), supplied by the workshop
teacher_review.md               supervisor feedback on the proposal; scope of the final report
data/                           dataset, untracked
examples/                       reference notebooks, untracked
outputs/                        generated artifacts, untracked except split.csv
docs/CODE_GUIDE.md              study and presentation guide
CLAUDE.md, .claude/             working rules and tooling
```

All paths used in the notebook are relative to the repository root.

## Environment

Runs in the `kaggle` conda environment on Python 3.14 with CUDA.

```bash
conda activate kaggle
pip install -r requirements.txt
jupyter lab
```

Verify the GPU is visible before running the training section:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Developed and executed on an NVIDIA RTX 3060 Laptop GPU (6 GB).

## Reproducing

Open `proposal.ipynb` and run Kernel > Restart and Run All. All randomness is seeded
(`SEED = 42`). The train/validation/test split is computed over all 458 images and
written to `outputs/split.csv`; that file is frozen so the final report inherits the
identical assignment.

Check that all authored text is plain ASCII:

```bash
python .claude/tools/check_ascii.py
```

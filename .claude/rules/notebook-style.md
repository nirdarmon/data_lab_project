# Rule: Notebook style

The notebook is the graded deliverable - a continuous document, not a pile of cells.

- Roughly 50% prose, 50% code. English only.
- No filler prose. Say what is computed, why over the alternative, what to conclude.
  Not "we now split the data" but "we split on duplicate groups because consecutive
  photos show the same wall".
- Define each term on first use (Dice, IoU, skip connection). Reader knows basic ML,
  not segmentation.
- Every figure gets an interpretation after it. A figure with no conclusion is
  deleted. Titles state the finding, not the mechanism.
- Segmentation results always render as: image | ground truth | prediction | overlay.
- Small single-purpose cells. Helpers defined once, near the top of the chapter that
  uses them.
- Restart-and-Run-All is the acceptance test. No hidden cross-cell state.
- One `SEED` constant drives all randomness.
- Print shapes and counts after transformations. Keep methodological assertions (no
  leakage, dimensions match) in the notebook with visible pass output - they are
  evidence.
- Cache expensive steps to `outputs/`; the cache key includes every parameter.
- Label every table and figure with how many images produced it. Subsets must never
  look broader than they are.
- Report failures and weak baselines; state which split every number came from.

# Rule: ML methodology non-negotiables

Breaking these invalidates the results. Dataset-specific, not style preferences.

1. **Split source images, never patches.** Patches from one photo are near-identical.
   Split on *near-duplicate groups* (campus photos repeat the same wall) - per-image
   is necessary but not sufficient. Every patch carries `source_image_id`; assert the
   three id sets are disjoint and print the pass.
2. **The split is frozen** in `outputs/split.csv` (committed, all 458 images). The
   final report loads it; it is never recomputed.
3. **Always `ImageOps.exif_transpose` the RGB.** 179 files are Orientation=6, 15 are
   Orientation=3; masks carry no rotation. The 15 are the trap - dimensions match
   either way while the image is upside-down, so shape equality is NOT proof of
   alignment. Verify alignment by content. Load RGB only through the one helper.
4. **Masks are lossy JPEG.** 1.104% of pixels are exactly 255 but 0.519% are 1-254 -
   one ambiguous pixel for every two crack pixels. Justify the binarization threshold
   from the histogram and report how many pixels it moves. Crack is white (255).
5. **Pixel accuracy is never the headline.** Crack ratio per image is mean 1.36%
   (median 1.17%, range 0.39-8.67%), so all-background scores ~98.9%. Report Dice and
   IoU; show accuracy once, to kill it.
6. **State the empty-mask convention** (Dice is 0/0 there, and most patches are
   empty). Report both micro and macro - they diverge sharply.
7. **Train-only statistics.** Normalization, thresholds, hyperparameters come from
   train/val. Test is touched once. Every number names its split.
8. **Never downscale whole images to patch size** - cracks are a few pixels wide.
   Crop at native resolution. Downscale-then-crop is an experiment, not the default.
9. **Deterministic val/test grids**: fixed, non-overlapping, no offsets, no
   augmentation. Random cropping and augmentation are train-only.
10. **Test distribution stays natural.** Training may rebalance negatives; val/test
    grids stay complete and unsampled.
11. **Segmentation is the only task.** RGB patch -> binary mask. No classification
    head. Mask statistics are descriptive only, never model inputs.

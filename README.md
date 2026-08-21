# Grace-Muller-Lab-2025

Image analysis code for confocal immunofluorescence stacks of cardiomyocytes. Two independent pipelines:
1. **Elongated cell edge vs. cytoplasm** -- filters to elongated (rod-shaped) cells and compares membrane-edge vs. cytoplasmic intensity.
2. **Cell pole/side (3D)** -- segments each cell in 3D and compares its pole ("End") vs. membrane ("Side") regions, including cross-channel colocalization.

## Environment

`Xana_env.yml` is the conda environment used to run everything here:

```
conda env create -f Xana_env.yml
conda activate Xana_env
```

## Contents

- **`Cell_pole_side_3D.ipynb`** (and the `_02`/`_03` copies) -- the 3D pole/side pipeline. For a folder of per-channel TIFF Z-stacks, it segments each cell in 3D (Otsu thresholding), finds its two poles, and builds per-Z-slice End (pole) and Side (edge) regions; batches this across every cell in the folder to measure per-channel intensity and colocalization (Pearson's r, Manders' M1/M2), exporting tidy CSVs, summary plots, per-cell overview images, and 3D TIFFs for viewing in ImageJ. The `_02`/`_03` copies are identical except for which dataset folder they point at (`FolderOI`), so multiple datasets can be processed side by side.
- **`Elongated_cell_edge_vs_cytoplasm.ipynb`**, **`_2.ipynb`**, **`_ColoRect.ipynb`** -- the elongated-cell pipeline: segments each cell, keeps only elongated (rod-shaped) ones by eccentricity / major-to-minor axis ratio, and measures per-channel intensity in each cell's membrane edge vs. its cytoplasmic interior, compared across conditions. The three copies point at different dataset folders (`FolderOI`).
- **`pole_membrane_intensity.py`** -- a standalone, pure-Python port of the original `Threshold_cells_and_define_edge` ImageJ macro (max-project, threshold, find poles by skeleton endpoints, measure End vs Side intensity) -- an early prototype of the ideas behind the 3D pole/side pipeline above, not the elongated-cell one. Run directly (`python pole_membrane_intensity.py`) to sanity-check the mask/skeleton/pole geometry against synthetic data, no real data required.

## Output layout

The 3D pipeline writes into two folders per dataset, alongside the input:
- `<dataset>_results/` -- CSVs, summary plots, small enough to share.
- `<dataset>_RepImages/` -- per-cell overview images, region QC plots, and 3D TIFFs; larger, kept separate from the shareable results.

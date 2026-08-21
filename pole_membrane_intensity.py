"""
Pure-Python replacement for the Threshold_cells_and_define_edge ImageJ macro.

Pipeline
--------
1. Max-Z-project the channels used for segmentation, normalize each by its
   own max and average them into one 8-bit image (mirrors the macro's
   "Normalize 3 channels ... average ... scale to 0-255").
2. Threshold that image (Otsu, standing in for the macro's Huang threshold --
   scikit-image has no Huang implementation), morphologically open it, and
   keep the single largest object (>= min_size) as "the cell" -- replaces
   "Convert to Mask" + "iterations=10 ... do=Open" + largest particle from
   the macro.
3. Rotate the whole stack so the cell's long axis is vertical (replaces the
   fitted-ellipse-angle rotation).
4. Skeletonize the rotated mask, find skeleton endpoints, and take the two
   endpoints that are furthest apart -> the cell's two "poles".
5. Build a small region around each pole and a band around the whole
   membrane (excluding the poles), then measure per-channel mean/sum
   intensity (sum projected across Z, DAPI excluded) in each region.
   This is the "End vs Side" comparison the macro set up but never filled in.

Requires: numpy, scipy, scikit-image, pandas.

Run this file directly (`python pole_membrane_intensity.py`) to sanity-check
the mask-building and rotation/skeleton/pole geometry against synthetic
data -- no real data needed for that check.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import (
    rotate as ndi_rotate,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    convolve,
)
from scipy.spatial.distance import pdist, squareform
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, disk


# ---------------------------------------------------------------------------
# 1. Build the image that will be thresholded/segmented
# ---------------------------------------------------------------------------

def normalize_and_average(channel_images):
    """Normalize each 2D channel image by its own max, average, scale to 0-255 uint8.

    channel_images : sequence of 2D arrays (same shape), e.g. max projections
        of the channels used to outline the cell.
    """
    norm_sum = None
    for img in channel_images:
        img = img.astype(np.float32)
        m = img.max()
        norm = img / m if m > 0 else img
        norm_sum = norm if norm_sum is None else norm_sum + norm

    averaged = norm_sum / len(channel_images)
    scaled = averaged * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Threshold segmentation -> largest object over min_size
# ---------------------------------------------------------------------------

def segment_by_threshold(image, open_iterations=10):
    """Threshold `image` and label connected components.

    Mirrors the macro's setAutoThreshold("Huang dark") + "Convert to Mask" +
    "Options... iterations=10 count=1 black do=Open". scipy's binary_opening
    with the default (cross-shaped, count=1-equivalent) structure and
    `iterations` repeats erosion then dilation `iterations` times, the same
    way ImageJ's Binary Options iterations do.
    """
    threshold = threshold_otsu(image)
    binary = image > threshold
    opened = binary_opening(binary, iterations=open_iterations)
    return label(opened)


def largest_label_mask(label_image, min_size):
    """Return a bool mask of the largest connected object with area >= min_size.

    Raises ValueError if no object meets min_size (mirrors the macro's
    "No ROI detected over the size of Min_size" exit).
    """
    props = regionprops(label_image)
    candidates = [p for p in props if p.area >= min_size]
    if not candidates:
        raise ValueError(f"No object with area >= {min_size} px found.")
    largest = max(candidates, key=lambda p: p.area)
    return label_image == largest.label


def clean_binary_mask(mask, open_radius=5):
    """Morphological open + fill holes (mirrors the macro's Options.../Open)."""
    opened = binary_opening(mask, structure=disk(open_radius))
    return binary_fill_holes(opened)


# ---------------------------------------------------------------------------
# 3. Rotate so the long axis is vertical
# ---------------------------------------------------------------------------

def mask_rotation_angle_deg(mask):
    """Degrees to rotate `mask` (via rotate_stack below) so its major axis is vertical."""
    props = regionprops(mask.astype(np.uint8))
    if not props:
        raise ValueError("Empty mask, cannot compute orientation.")
    # skimage orientation: angle between row-axis (vertical) and the major
    # axis, in (-pi/2, pi/2]. It is already 0 when the major axis is
    # vertical, so rotating by -orientation straightens it.
    orientation_rad = props[0].orientation
    return -np.degrees(orientation_rad)


def rotate_stack(stack, angle_deg, order=1, reshape=True):
    """Rotate the last two axes of `stack` (2D mask or N-D intensity stack)
    by angle_deg, counter-clockwise, padding with 0.
    """
    return ndi_rotate(
        stack, angle_deg, axes=(-2, -1), reshape=reshape,
        order=order, mode="constant", cval=0,
    )


# ---------------------------------------------------------------------------
# 4. Skeleton endpoints -> poles
# ---------------------------------------------------------------------------

_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def skeleton_endpoints(mask):
    """Skeletonize `mask`, return (skeleton, endpoint_coords[N,2] as (row, col))."""
    skeleton = skeletonize(mask)
    neighbor_count = convolve(skeleton.astype(int), _NEIGHBOR_KERNEL, mode="constant")
    endpoints = skeleton & (neighbor_count == 1)
    coords = np.column_stack(np.nonzero(endpoints))
    return skeleton, coords


def furthest_pair(coords):
    """Given (row, col) points, return the two furthest apart and their distance."""
    if len(coords) < 2:
        raise ValueError("Need at least 2 endpoints to find a furthest pair.")
    dists = squareform(pdist(coords))
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    return coords[i], coords[j], dists[i, j]


# ---------------------------------------------------------------------------
# 5. Pole regions + membrane band, and intensity measurement
# ---------------------------------------------------------------------------

def pole_and_membrane_regions(mask, pole1, pole2, end_radius=100, side_band=10):
    """Build two pole-cap regions and a membrane band, all clipped to `mask`.

    end_radius : pixels from each pole counted as "end" (mirrors the macro's
        100-iteration dilation around each endpoint, clipped to the cell).
    side_band  : pixels in from the mask boundary counted as "membrane"
        (mirrors the macro's 10-iteration dilation of the outline).
    Side excludes anything already claimed by either end region.
    """
    yy, xx = np.indices(mask.shape)
    dist1 = np.hypot(yy - pole1[0], xx - pole1[1])
    dist2 = np.hypot(yy - pole2[0], xx - pole2[1])

    end1 = mask & (dist1 <= end_radius)
    end2 = mask & (dist2 <= end_radius)

    boundary_dist = distance_transform_edt(mask)
    side = mask & (boundary_dist <= side_band) & ~end1 & ~end2

    return {"Upper-end": end1, "Lower-end": end2, "Sides": side}


def measure_regions(stack, regions, channel_names=None, dapi_channel=0):
    """stack: (C, Z, Y, X) intensity array. regions: name -> bool mask (Y, X).

    Sum-projects each non-DAPI channel across Z, then reports mean/sum
    intensity per region. Returns a tidy DataFrame.
    """
    n_channels = stack.shape[0]
    if channel_names is None:
        channel_names = [f"C{c}" for c in range(n_channels)]

    rows = []
    for c in range(n_channels):
        if c == dapi_channel:
            continue
        proj = stack[c].sum(axis=0)
        for region_name, region_mask in regions.items():
            area = int(region_mask.sum())
            if area == 0:
                mean_val, sum_val = np.nan, np.nan
            else:
                pixels = proj[region_mask]
                mean_val, sum_val = float(pixels.mean()), float(pixels.sum())
            rows.append({
                "Channel": channel_names[c],
                "Region": region_name,
                "Area": area,
                "Mean": mean_val,
                "Sum": sum_val,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def analyze_cell_poles_vs_membrane(
    stack,
    dapi_channel=0,
    seg_channels=(0, 1, 2),
    min_size=20000,
    open_iterations=10,
    open_radius=5,
    end_radius=100,
    side_band=10,
    channel_names=None,
):
    """stack: (C, Z, Y, X) numpy array for one field of view.

    Returns a dict with every intermediate artifact plus the final
    "results" DataFrame (Channel x Region -> Area/Mean/Sum), so the caller
    can QC each stage (mask, rotated stack, skeleton, poles, regions).
    """
    # 1. max project + normalize/average for segmentation input
    max_proj = stack.max(axis=1)  # (C, Y, X)
    seg_image = normalize_and_average([max_proj[c] for c in seg_channels])

    # 2. threshold, open, keep the largest object over min_size
    labels = segment_by_threshold(seg_image, open_iterations=open_iterations)
    mask = largest_label_mask(labels, min_size)

    # 3. rotate stack + mask so the long axis is vertical
    angle = mask_rotation_angle_deg(mask)
    rotated_stack = rotate_stack(stack, angle, order=1, reshape=True)
    rotated_mask = rotate_stack(mask.astype(np.uint8), angle, order=0, reshape=True) > 0
    rotated_mask = clean_binary_mask(rotated_mask, open_radius=open_radius)

    # 4. skeleton + poles
    skeleton, endpoint_coords = skeleton_endpoints(rotated_mask)
    pole1, pole2, pole_distance = furthest_pair(endpoint_coords)

    # 5. regions + measurement
    regions = pole_and_membrane_regions(
        rotated_mask, pole1, pole2, end_radius=end_radius, side_band=side_band,
    )
    results = measure_regions(
        rotated_stack, regions, channel_names=channel_names, dapi_channel=dapi_channel,
    )

    return {
        "seg_image": seg_image,
        "mask": mask,
        "rotation_angle_deg": angle,
        "rotated_stack": rotated_stack,
        "rotated_mask": rotated_mask,
        "skeleton": skeleton,
        "endpoint_coords": endpoint_coords,
        "pole1": pole1,
        "pole2": pole2,
        "pole_distance": pole_distance,
        "regions": regions,
        "results": results,
    }


# ---------------------------------------------------------------------------
# QC plotting
# ---------------------------------------------------------------------------

def plot_qc(result):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(result["rotated_mask"], cmap="gray")
    ys, xs = result["endpoint_coords"][:, 0], result["endpoint_coords"][:, 1]
    axes[0].imshow(np.ma.masked_where(~result["skeleton"], result["skeleton"]),
                    cmap="autumn", alpha=0.8)
    axes[0].scatter(xs, ys, s=10, c="cyan")
    axes[0].scatter([result["pole1"][1], result["pole2"][1]],
                     [result["pole1"][0], result["pole2"][0]],
                     s=80, marker="x", c="red")
    axes[0].set_title(f"Skeleton + poles (rotated {result['rotation_angle_deg']:.1f} deg)")

    overlay = np.zeros(result["rotated_mask"].shape + (3,), dtype=float)
    overlay[result["regions"]["Sides"]] = (0, 1, 0)
    overlay[result["regions"]["Upper-end"]] = (1, 0, 0)
    overlay[result["regions"]["Lower-end"]] = (0, 0, 1)
    axes[1].imshow(overlay)
    axes[1].set_title("Regions: red=upper end, blue=lower end, green=sides")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Self-test with synthetic data (no real data needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from skimage.draw import ellipse

    # --- 1+2: normalize/average + threshold + largest-object mask building ---
    rng = np.random.default_rng(0)
    canvas = np.zeros((300, 300), dtype=bool)
    rr, cc = ellipse(150, 150, 40, 100, rotation=np.radians(30))
    canvas[rr, cc] = True

    channel_images = []
    for peak in (200.0, 150.0, 220.0):
        img = rng.normal(loc=5, scale=2, size=canvas.shape).clip(min=0)
        img[canvas] += peak
        channel_images.append(img)

    seg_image = normalize_and_average(channel_images)
    labels = segment_by_threshold(seg_image, open_iterations=5)
    mask = largest_label_mask(labels, min_size=1000)
    overlap = (mask & canvas).sum() / canvas.sum()
    print(f"mask-building: recovered {mask.sum()} px, {overlap:.1%} overlap with ground truth")
    assert overlap > 0.9, "threshold-based mask building did not recover the synthetic cell"

    try:
        largest_label_mask(labels, min_size=10**9)
        raise AssertionError("largest_label_mask should have raised for an impossible min_size")
    except ValueError:
        pass

    # --- 3-5: rotation / skeleton / pole geometry ---
    angle = mask_rotation_angle_deg(canvas)
    rotated = rotate_stack(canvas.astype(np.uint8), angle, order=0, reshape=True) > 0
    rotated = clean_binary_mask(rotated, open_radius=3)

    check_angle = mask_rotation_angle_deg(rotated)
    print(f"applied rotation: {angle:.2f} deg, residual orientation after rotation: {check_angle:.2f} deg")
    assert abs(check_angle) < 2, "rotation did not straighten the major axis"

    skeleton, endpoints = skeleton_endpoints(rotated)
    p1, p2, dist = furthest_pair(endpoints)
    print(f"found {len(endpoints)} endpoints, furthest pair distance = {dist:.1f} px")
    # major axis of the ellipse is 200 px long (2 * 100), poles should be close to that
    assert 150 < dist < 220, "pole distance is way off for a 200px-long ellipse"

    regions = pole_and_membrane_regions(rotated, p1, p2, end_radius=40, side_band=8)
    for name, m in regions.items():
        print(f"{name}: {m.sum()} px")

    print("Self-test passed.")

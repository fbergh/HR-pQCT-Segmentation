import argparse
import os
import warnings
from pathlib import Path
from typing import Set, List, Tuple, Callable, Dict, Any

import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

from compare import shift_masks, compute_diff, load_aim
from utils.segmentation_evaluation import binarize_numpy_array, get_distance_map_and_surface


def parse_slices(all_slices: Any) -> List[int]:
    if all_slices is None:
        return all_slices
    if all_slices:
        if type(all_slices) == list:
            return [int(x) for x in all_slices]
        try:
            return [int(all_slices)]
        except:
            return list(range(0, 168, 1))
    else:
        return [60]


def _globify(s: str) -> str:
    return "".join(f"[{l.lower()}{l.upper()}]" for l in s)


def gather_paths_for_ids(data_dir: Path, filenames: Set[str]) -> List[Path]:
    return sorted(
        file.resolve()
        for file in data_dir.iterdir()
        if file.is_file() and file.name.lower() in filenames
    )


def gather_scan_paths_for_ids(data_dir: Path, scan_ids: List[str], scan_type: str) -> List[Path]:
    suffix = ""
    if "fxmovie" in scan_type:
        suffix = "_crop"
    if not scan_ids:
        return sorted(data_dir.glob(f"[cC]*[0-9]{_globify(suffix)}.[aA][iI][mM]"))
    target_filenames = {f"{scan_id.lower()}{suffix}.aim" for scan_id in scan_ids}
    return gather_paths_for_ids(data_dir, target_filenames)


def gather_mask_paths_for_ids(mask_dir: Path, scan_ids: List[str], scan_type: str) -> Tuple[List[Path], List[Path]]:
    def _replace(path: Path, scan_type_: str):
        fn = path.stem
        extension = path.suffix
        if "fxmovie_acon" in scan_type_:
            fn = fn.replace("trmsk", "crmsk")
        else:
            fn = fn.replace("trab", "cort").replace("TRAB", "CORT")
        return Path(path.parent, f"{fn}{extension}")

    # Determine the suffix for the filename
    suffix = "_trab_mask"
    if "fxmovie" in scan_type and "pred" in scan_type:
        suffix = "_crop_trab_mask"
    elif "fxmovie_acon" in scan_type:
        suffix = "_trmsk_a"
    elif "fxmovie_scon" in scan_type:
        suffix = "_trab_mask_scon"
    elif "xtremect1_periosteal" in scan_type and "pred" not in scan_type:
        suffix = "_mask"

    if not scan_ids:
        trab_mask_paths = sorted(mask_dir.glob(f"[cC]*[0-9]{_globify(suffix)}.[aA][iI][mM]"))
    else:
        target_trab_filenames = {f"{scan_id.lower()}{suffix}.aim" for scan_id in scan_ids}
        trab_mask_paths = gather_paths_for_ids(mask_dir, target_trab_filenames)
    cort_mask_paths = [_replace(p, scan_type) for p in trab_mask_paths]
    return trab_mask_paths, cort_mask_paths


def sync_paths_by_common_ids(
        path_lists: List[List[Path]],
        extract_id: Callable[[Path], str] = lambda p: p.stem
) -> List[List[Path]]:
    """
    Filters multiple lists of Path objects so they only contain files with IDs
    common to all lists. The result preserves the same order across lists.

    :param path_lists: A list of lists, each containing Path objects from different directories.
    :param extract_id: Function to extract the ID from each Path (default uses stem).
    :return: A list of filtered lists, one per input list, all with the same IDs in matching order.
    """
    id_maps = []
    id_sets = []

    # Map each ID to its corresponding Path for all lists
    for paths in path_lists:
        id_map = {}
        for p in paths:
            id_ = extract_id(p)
            if id_:
                id_map[id_] = p
        id_maps.append(id_map)
        id_sets.append(set(id_map.keys()))

    # Find the intersection of IDs across all lists
    common_ids = set.intersection(*id_sets)

    # Sort the common IDs to ensure consistent order
    sorted_common_ids = sorted(common_ids)

    for i, path_type in enumerate(["scans", "pred_trab", "pred_cort", "ref_trab", "ref_cort"]):
        missing_ids = set(sorted_common_ids) - id_sets[i]
        if len(missing_ids) > 0:
            warnings.warn(f"{path_type} is missing scan ids: {sorted(missing_ids)}")
        superfluous_ids = id_sets[i] - set(sorted_common_ids)
        if len(superfluous_ids) > 0:
            warnings.warn(f"{path_type} has superfluous scan ids: {sorted(superfluous_ids)}")

    # Build filtered lists in matching order
    filtered_lists = [
        [id_map[id_] for id_ in sorted_common_ids]
        for id_map in id_maps
    ]

    return filtered_lists


def load_masks(trab_mask_path: str, cort_mask_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    trab_mask, trab_md = load_aim(trab_mask_path)
    cort_mask, cort_md = load_aim(cort_mask_path)
    return dict(trab_mask=(trab_mask > 0).astype(float), cort_mask=(cort_mask > 0).astype(float)), \
        dict(trab_md=trab_md, cort_md=cort_md)


def plot_four_part_image(scan, pred, ref, slice_idx, scan_id, output_path, show=False, save=True):
    pred = {k: v[..., slice_idx] for k, v in pred.items()}
    ref = {k: v[..., slice_idx] for k, v in ref.items()}
    slice_ = scan[..., slice_idx]

    fig, ax = plt.subplots(2, 2, figsize=(20, 12))

    # Plot original image
    ax[0, 0].imshow(slice_, cmap='gray')
    ax[0, 0].set_title("Original image")
    ax[0, 0].set_axis_off()

    # Plot img with gt masks
    ax[1, 0].imshow(slice_, cmap='gray')
    ax[1, 0].imshow(ref["trab_mask"], cmap='Blues', alpha=0.7 * ref["trab_mask"])
    ax[1, 0].imshow(ref["cort_mask"], cmap='Greens', alpha=0.7 * ref["cort_mask"])
    ax[1, 0].set_title("Reference")
    ax[1, 0].set_axis_off()

    # Plot img with pred masks
    ax[1, 1].imshow(slice_, cmap='gray')
    ax[1, 1].imshow(pred["trab_mask"], cmap='Blues', alpha=0.7 * pred["trab_mask"])
    ax[1, 1].imshow(pred["cort_mask"], cmap='Greens', alpha=0.7 * pred["cort_mask"])
    ax[1, 1].set_title("Prediction (by AI model)")
    ax[1, 1].set_axis_off()

    # Plot img with overlay
    ax[0, 1].imshow(slice_, cmap='gray')
    ax[0, 1].imshow(pred["trab_mask"], cmap='Blues', alpha=0.7 * pred["trab_mask"])
    ax[0, 1].imshow(pred["cort_mask"], cmap='Greens', alpha=0.7 * pred["cort_mask"])
    diff = compute_diff(pred, ref)
    ax[0, 1].imshow(diff, cmap='Reds', alpha=0.7 * diff)
    ax[0, 1].set_title("Segmentation overlay (red = difference)")
    ax[0, 1].set_axis_off()

    plt.suptitle(f"Four-part image of scan, reference, prediction, and overlay "
                 f"(slice index={slice_idx} & scan id={scan_id})")
    if save:
        plt.savefig(Path(output_path, f"{scan_id}_fourpart_{slice_idx}.png"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(plt.gcf())


def find_extreme_hausdorff_slices(seg, ref, spacing, percentage=0.95):
    # binarize reference and segmentation
    ref, seg = binarize_numpy_array(ref), binarize_numpy_array(seg)

    # convert numpy binary matrices into binary SITK images
    ref = sitk.GetImageFromArray(ref)
    seg = sitk.GetImageFromArray(seg)
    ref.SetSpacing([spacing] * ref.GetDimension())
    seg.SetSpacing([spacing] * seg.GetDimension())

    # calculate the surface and distance maps for reference and segmentation
    ref_dist_map, ref_surface, ref_surface_num_pix = get_distance_map_and_surface(ref)
    seg_dist_map, seg_surface, seg_surface_num_pix = get_distance_map_and_surface(seg)

    # get the symmetric distances by multiplying the reference distance map by
    # the segmentation surface and vice versa
    seg2ref_dist_map = ref_dist_map * sitk.Cast(seg_surface, sitk.sitkFloat32)
    ref2seg_dist_map = seg_dist_map * sitk.Cast(ref_surface, sitk.sitkFloat32)

    seg2ref_dist_map_np = sitk.GetArrayFromImage(seg2ref_dist_map)
    ref2seg_dist_map_np = sitk.GetArrayFromImage(ref2seg_dist_map)
    seg2ref_max = seg2ref_dist_map_np.max()
    ref2seg_max = ref2seg_dist_map_np.max()
    largest_dist_map = ref2seg_dist_map_np if ref2seg_max > seg2ref_max else seg2ref_dist_map_np
    hausdorff = max(ref2seg_max, seg2ref_max)

    _, _, large_err_slices = (largest_dist_map > hausdorff * percentage).nonzero()
    return large_err_slices.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_gt_dir")
    parser.add_argument("path_to_pred_dir")
    parser.add_argument("output_path")
    parser.add_argument("scan_type", choices=["fxmovie_acon", "fxmovie_scon", "xtremect1_cort_trab",
                                              "xtremect1_periosteal"])
    # Optional arguments
    parser.add_argument("--path_to_ref_masks_dir", default=None)
    parser.add_argument("--filenames", nargs='*', default=None)
    parser.add_argument("--slices", default=None)
    parser.add_argument("--extreme_hausdorff_slices", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    max_n_slices = 10
    slices = args.slices
    if slices:
        slices = parse_slices(slices)
        max_n_slices = len(slices)

    extreme_hausdorff_percentage = 0.7

    scan_paths = gather_scan_paths_for_ids(Path(args.path_to_gt_dir), args.filenames, args.scan_type)
    pred_paths = gather_mask_paths_for_ids(Path(args.path_to_pred_dir), args.filenames, args.scan_type + "_pred")
    ref_masks_dir = args.path_to_ref_masks_dir if args.path_to_ref_masks_dir is not None else args.path_to_gt_dir
    ref_paths = gather_mask_paths_for_ids(Path(ref_masks_dir), args.filenames, args.scan_type)

    all_paths = sync_paths_by_common_ids([scan_paths, *pred_paths, *ref_paths], extract_id=lambda p: p.stem[:8])
    zipped_paths = zip(*all_paths)

    for i, (scan_path, pred_trab_path, pred_cort_path, ref_trab_path, ref_cort_path) in enumerate(zipped_paths):
        scan_id = scan_path.stem[:8]
        print(f"{i + 1}/{len(all_paths[0])} - {scan_id}", flush=True, end=" ")
        scan, scan_md = load_aim(str(scan_path))
        scan = scan[1:, 1:]
        pred, pred_md = load_masks(str(pred_trab_path), str(pred_cort_path))
        ref, ref_md = load_masks(str(ref_trab_path), str(ref_cort_path))

        # Linear registration to ensure pred and gt are in the same location
        pred_sample = {"cort_mask_position": pred_md["cort_md"]["position"],
                       "trab_mask_position": pred_md["trab_md"]["position"],
                       "image_position": scan_md["position"],
                       "image": scan, **pred}
        ref_sample = {"cort_mask_position": ref_md["cort_md"]["position"],
                      "trab_mask_position": ref_md["trab_md"]["position"],
                      "image_position": scan_md["position"],
                      "image": scan, **ref}
        scan, pred, ref = shift_masks(pred_sample, ref_sample)

        if args.extreme_hausdorff_slices:
            print("- Searching for Hausdorff slices -", end=" ")
            spacing = 8.2e-5 if "xtremect1" in args.scan_type else 6.1e-5
            trab_slices = find_extreme_hausdorff_slices(pred["trab_mask"], ref["trab_mask"], spacing,
                                                        percentage=extreme_hausdorff_percentage)
            cort_slices = find_extreme_hausdorff_slices(pred["cort_mask"], ref["cort_mask"], spacing,
                                                        percentage=extreme_hausdorff_percentage)
            slices = np.asarray(list(set(trab_slices + cort_slices)))
            if len(slices) > max_n_slices:
                slices = slices[np.linspace(0, len(slices) - 1, max_n_slices, endpoint=True, dtype=int)]

        for slice_idx in slices:
            print(slice_idx, end=" ")
            plot_four_part_image(scan, pred, ref, slice_idx, scan_id, args.output_path, show=False, save=True)
        print()


if __name__ == "__main__":
    main()

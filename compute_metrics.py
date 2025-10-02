"""

1. Get all prediction AIM paths
2. Load and preprocess prediction AIM
3. Find corresponding reference AIM
4. Load and preprocess reference AIM
5. Compute metric
6. Enter in dataframe

"""
import argparse
import math
import warnings
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from test_grounds import load_aim, create_sample, sample_padder
from utils.segmentation_evaluation import calculate_dice_and_jaccard, calculate_surface_distance_measures

EXCEPTIONS = []


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir")
    parser.add_argument("--scan_pattern")
    parser.add_argument("--reference_dir")
    parser.add_argument("--reference_cort_pattern")
    parser.add_argument("--reference_trab_pattern")
    parser.add_argument("--prediction_dir")
    parser.add_argument("--save_dir")
    return parser


def _get_scan_id(p, do_lower=True):
    fn = p.stem
    if do_lower:
        fn = fn.lower()
    if '_' in fn:
        fn = fn[:fn.index('_')]
    return fn


def filter_paths(data_paths, ref_cort_paths, ref_trab_paths, pred_paths):
    def _filter_paths(filter_ids, filter_paths, pred_ids_):
        desired_filter_ids = sorted(set(filter_ids).intersection(pred_ids_))
        for i in reversed(range(len(filter_ids))):
            if filter_ids[i] not in desired_filter_ids:
                filter_paths.pop(i)
        return filter_paths
    data_ids = [_get_scan_id(p) for p in data_paths]
    ref_cort_ids = [_get_scan_id(p) for p in ref_cort_paths]
    ref_trab_ids = [_get_scan_id(p) for p in ref_trab_paths]
    pred_ids = [_get_scan_id(p) for p in pred_paths]

    # Check if the data files are missing any files for which we have predictions. If so, remove them from the predictions
    # For which ids do we have predictions, but no data? Remove those predictions.
    to_remove = set([x.lower() for x in EXCEPTIONS])
    if len(set(pred_ids) - set(data_ids)) > 0:
        warnings.warn(f"Scan list is missing: {sorted(set(pred_ids) - set(data_ids))}. Removing these.")
        to_remove.update(set(pred_ids) - set(data_ids))
    if len(set(pred_ids) - set(ref_cort_ids)) > 0:
        warnings.warn(f"Reference cort list is missing: {sorted(set(pred_ids) - set(ref_cort_ids))}. Removing these.")
        to_remove.update(set(pred_ids) - set(ref_cort_ids))
    if len(set(pred_ids) - set(ref_trab_ids)) > 0:
        warnings.warn(f"Reference trab list is missing: {sorted(set(pred_ids) - set(ref_trab_ids))}. Removing these.")
        to_remove.update(set(pred_ids) - set(ref_trab_ids))

    for scan_id in sorted(to_remove, reverse=True):
        for paths, ids in [(pred_paths, pred_ids), (data_paths, data_ids),
                           (ref_cort_paths, ref_cort_ids), (ref_trab_paths, ref_trab_ids)]:
            if scan_id in ids:
                paths.pop(ids.index(scan_id))
                ids.remove(scan_id)

    # Check if the prediction dir is missing some data files. If so, remove them.
    # For which ids do we have data, but no predictions? Remove those data.
    if len(set(data_ids) - set(pred_ids)) > 0:
        warnings.warn(f"Pred ids does not contain the following scan ids: {sorted(set(data_ids) - set(pred_ids))}")
    if len(set(ref_cort_ids) - set(pred_ids)) > 0:
        warnings.warn(f"Pred ids does not contain the following ref cort ids: {sorted(set(ref_cort_ids) - set(pred_ids))}")
    if len(set(ref_trab_ids) - set(pred_ids)) > 0:
        warnings.warn(f"Pred ids does not contain the following ref trab ids: {sorted(set(ref_trab_ids) - set(pred_ids))}")

    data_paths = _filter_paths(data_ids, data_paths, pred_ids)
    ref_cort_paths = _filter_paths(ref_cort_ids, ref_cort_paths, pred_ids)
    ref_trab_paths = _filter_paths(ref_trab_ids, ref_trab_paths, pred_ids)

    return list(zip(data_paths, ref_cort_paths, ref_trab_paths, pred_paths))


def compute_metrics(pred_cort, ref_cort, pred_trab, ref_trab, spacing, to_mm=True):
    v = dict()
    v["dice_cort"], v["jaccard_cort"] = calculate_dice_and_jaccard(pred_cort, ref_cort)
    v["dice_trab"], v["jaccard_trab"] = calculate_dice_and_jaccard(pred_trab, ref_trab)
    ssd_cort = calculate_surface_distance_measures(pred_cort, ref_cort, [spacing, spacing, spacing])
    v["ssd_max_cort"], v["ssd_mean_cort"] = ssd_cort['max'], ssd_cort['mean']
    ssd_trab = calculate_surface_distance_measures(pred_trab, ref_trab, [spacing, spacing, spacing])
    v["ssd_max_trab"], v["ssd_mean_trab"] = ssd_trab['max'], ssd_trab['mean']
    if to_mm:
        v["ssd_max_cort"] *= 1_000
        v["ssd_mean_cort"] *= 1_000
        v["ssd_max_trab"] *= 1_000
        v["ssd_mean_trab"] *= 1_000
    return v


def load_data(data_path, ref_cort_path, ref_trab_path, pred_path):
    def _pad(ref_sample_, pred_sample_):
        ref_sample_ = sample_padder(ref_sample_, 8)
        scan_ = ref_sample_['image']
        scan_md_ = dict(position=ref_sample_['image_position'])
        ref_cort_ = ref_sample_['cort_mask']
        ref_trab_ = ref_sample_['trab_mask']
        pred_cort_md_ = dict(position=pred_sample_["cort_mask_position"])
        pred_trab_md_ = dict(position=pred_sample_["trab_mask_position"])
        pred_sample_ = create_sample(scan_, scan_md_, pred_sample_["cort_mask"], pred_cort_md_, pred_sample_["trab_mask"],
                                     pred_trab_md_)
        pred_sample_ = sample_padder(pred_sample_, 8, padded_size=scan_.shape[0])
        return pred_sample_['image'], ref_cort_, ref_trab_, pred_sample_['cort_mask'], pred_sample_['trab_mask']

    scan, scan_md = load_aim(str(data_path), True)
    ref_cort, ref_cort_md = load_aim(str(ref_cort_path), False)
    ref_trab, ref_trab_md = load_aim(str(ref_trab_path), False)
    pred_cort, pred_cort_md = load_aim(str(pred_path), False)
    pred_trab, pred_trab_md = load_aim(str(pred_path).replace("CORT", "TRAB"), False)
    n_slices = 110 if "xtremect1" in str(data_path) else 168
    if scan.shape[-1] != n_slices:
        raise ValueError(f"Scan doesn't have {n_slices} slices")
    if ref_cort.shape[-1] != n_slices:
        raise ValueError(f"Reference cortical mask doesn't have {n_slices} slices")
    if ref_trab.shape[-1] != n_slices:
        raise ValueError(f"Reference trabecular mask doesn't have {n_slices} slices")
    ref_sample = create_sample(scan, scan_md, ref_cort, ref_cort_md, ref_trab, ref_trab_md)
    pred_sample = create_sample(scan, scan_md, pred_cort, pred_cort_md, pred_trab, pred_trab_md)
    try:
        scan, ref_cort, ref_trab, pred_cort, pred_trab = _pad(ref_sample, pred_sample)
    except:
        scan, ref_cort, ref_trab, pred_cort, pred_trab = _pad(pred_sample, ref_sample)
    return scan, ref_cort, ref_trab, pred_cort, pred_trab


def main():
    parser = create_parser()
    args = parser.parse_args()
    data_paths = sorted(Path(args.scan_dir).glob(args.scan_pattern))
    ref_cort_aim_paths = sorted(Path(args.reference_dir).glob(args.reference_cort_pattern))
    ref_trab_aim_paths = sorted(Path(args.reference_dir).glob(args.reference_trab_pattern))
    pred_aim_paths_raw = sorted(Path(args.prediction_dir, "raw").glob("*CORT_MASK.AIM"))
    pred_aim_paths_pp = sorted(Path(args.prediction_dir, "pp").glob("*CORT_MASK.AIM"))
    paths_raw = filter_paths(data_paths, ref_cort_aim_paths, ref_trab_aim_paths, pred_aim_paths_raw)
    paths_pp = filter_paths(data_paths, ref_cort_aim_paths, ref_trab_aim_paths, pred_aim_paths_pp)
    spacing = 8.2e-5 if "xtremect1" in args.scan_dir else 6.1e-5

    metrics = {"name": []}
    metrics.update({f"{metric}_{mask}_{processing}": [] for metric in ["dice", "jaccard", "ssd_max", "ssd_mean"]
                    for mask in ["cort", "trab"] for processing in ["raw", "post"]})

    for path_raw, path_pp in zip(paths_raw, paths_pp):
        name = _get_scan_id(path_raw[0], do_lower=False)
        print(name)
        metrics['name'].append(name)
        # Compute raw metrics
        try:
            scan, ref_cort, ref_trab, pred_cort, pred_trab = load_data(*path_raw)
            cur_metrics_raw = compute_metrics(pred_cort, ref_cort, pred_trab, ref_trab, spacing)
            for k, v in cur_metrics_raw.items():
                metrics[f"{k}_raw"].append(v)
        except ValueError as e:
            for metric in metrics.keys():
                if "raw" in metric:
                    metrics[metric].append(math.nan)
            print(e)
        # Compute postprocessing metrics
        try:
            scan, ref_cort, ref_trab, pred_cort, pred_trab = load_data(*path_pp)
            cur_metrics_post = compute_metrics(pred_cort, ref_cort, pred_trab, ref_trab, spacing)
            for k, v in cur_metrics_post.items():
                metrics[f"{k}_post"].append(v)
        except ValueError as e:
            for metric in metrics.keys():
                if "post" in metric:
                    metrics[metric].append(math.nan)
            print(e)

    output_path = Path(args.save_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metric_df = pd.DataFrame.from_dict(metrics)
    metric_df.to_csv(Path(output_path, "metrics2.csv"), index=False)


if __name__ == "__main__":
    main()

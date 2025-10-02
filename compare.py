import argparse
import math
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import vtkbone
from bonelab.util.aim_calibration_header import get_aim_density_equation
from skimage.measure import label as sklabel
from vtkmodules.util.numpy_support import vtk_to_numpy

TRAB_MASK_NAME_CAPS = "TRAB_MASK"
CORT_MASK_NAME_CAPS = "CORT_MASK"
TRAB_MASK_NAME = "trab_mask"
CORT_MASK_NAME = "cort_mask"


def load_aim(path_to_aim, to_density=False):
    reader = vtkbone.vtkboneAIMReader()
    reader.DataOnCellsOff()
    reader.SetFileName(path_to_aim)
    reader.Update()

    vtk_data = reader.GetOutput().GetPointData().GetScalars()
    data = vtk_to_numpy(vtk_data).reshape(reader.GetOutput().GetDimensions(), order='F')

    metadata = dict()
    metadata["processing_log"] = reader.GetProcessingLog()
    metadata["position"] = list(reader.GetPosition())
    metadata["spacing"] = reader.GetOutput().GetSpacing()
    metadata["origin"] = reader.GetOutput().GetOrigin()
    if to_density:
        m, b = get_aim_density_equation(reader.GetProcessingLog())
        return m * data + b, metadata
    else:
        return data, metadata


def load_scan(path_to_dir, scan_id):
    def load_scan_path(path_to_dir_, scan_id_):
        return_str = "nothing"
        if Path(path_to_dir_, f"{scan_id_}.aim").exists():
            return_str = str(Path(path_to_dir_, f"{scan_id_}.aim"))
        if Path(path_to_dir_, f"{scan_id_}.AIM").exists():
            return_str = str(Path(path_to_dir_, f"{scan_id_}.AIM"))
        if Path(path_to_dir_, f"{scan_id_}_crop.aim").exists():
            return_str = str(Path(path_to_dir_, f"{scan_id_}_crop.aim"))
        if Path(path_to_dir_, f"{scan_id_}_crop.AIM").exists():
            return_str = str(Path(path_to_dir_, f"{scan_id_}_crop.AIM"))
        return return_str

    scan_path = load_scan_path(path_to_dir, scan_id)
    scan, scan_md = load_aim(scan_path)
    return scan, scan_md


def load_masks(path_to_dir, scan_id, suffix=""):
    def load_mask_path(path_to_dir_, scan_id_, mask_type, suffix=""):
        caps_name = TRAB_MASK_NAME_CAPS if mask_type == "trab" else CORT_MASK_NAME_CAPS
        name = TRAB_MASK_NAME if mask_type == "trab" else CORT_MASK_NAME
        return_str = "nothing"
        # {scanid}_crop_{MASK_NAME}{suffix}.AIM
        if Path(path_to_dir_, f"{scan_id_}_crop_{caps_name}{suffix}.AIM").exists():
            return_str = str(Path(path_to_dir_, f"{scan_id_}_crop_{caps_name}{suffix}.AIM"))
        # {scanid}_{MASK_NAME}{suffix}.aim
        if Path(path_to_dir_, f"{scan_id_}_{caps_name}{suffix}.AIM").exists():
            return_str = str(Path(path_to_dir_, f"{scan_id_}_{caps_name}{suffix}.AIM"))
        # {scanid}_crop_{mask_name}{suffix}.aim
        if Path(path_to_dir_, f"{scan_id_}_crop_{name}{suffix.lower()}.aim").exists():
            return_str = str(Path(path_to_dir_, f"{scan_id_}_crop_{name}{suffix.lower()}.aim"))
        # {scanid}_{mask_name}{suffix}.aim
        if Path(path_to_dir_, f"{scan_id_}_{name}{suffix.lower()}.aim").exists():
            return_str = str(Path(path_to_dir_, f"{scan_id_}_{name}{suffix.lower()}.aim"))
        return return_str

    if "acon" in suffix:
        try:
            trab_path = str(Path(path_to_dir, f"{scan_id}_trmsk_a.aim"))
            cort_path = str(Path(path_to_dir, f"{scan_id}_crmsk_a.aim"))
            trab_mask, trab_md = load_aim(trab_path)
            cort_mask, cort_md = load_aim(cort_path)
        except:
            trab_path = load_mask_path(path_to_dir, scan_id, 'trab')
            cort_path = load_mask_path(path_to_dir, scan_id, 'cort')
            trab_mask, trab_md = load_aim(trab_path)
            cort_mask, cort_md = load_aim(cort_path)
    elif "periosteal" in suffix:
        cort_mask, cort_md = load_aim(str(Path(path_to_dir, f"{scan_id}_MASK.AIM")))
        trab_mask = cort_mask.copy()
        trab_md = deepcopy(cort_md)
    else:
        # Try with suffix. If it doesn't work, try without
        try:
            trab_path = load_mask_path(path_to_dir, scan_id, 'trab', suffix=suffix)
            cort_path = load_mask_path(path_to_dir, scan_id, 'cort', suffix=suffix)
            trab_mask, trab_md = load_aim(trab_path)
            cort_mask, cort_md = load_aim(cort_path)
        except:
            trab_path = load_mask_path(path_to_dir, scan_id, 'trab')
            cort_path = load_mask_path(path_to_dir, scan_id, 'cort')
            trab_mask, trab_md = load_aim(trab_path)
            cort_mask, cort_md = load_aim(cort_path)
    return dict(trab_mask=(trab_mask > 0).astype(float), cort_mask=(cort_mask > 0).astype(float)), \
        dict(trab_md=trab_md, cort_md=cort_md)


def to_shape(a, shape):
    y_, x_, _ = shape
    y, x, _ = a.shape
    y_pad = y_ - y
    x_pad = x_ - x

    if x_pad > 0 and y_pad > 0:
        return np.pad(a, ((y_pad // 2, y_pad // 2 + y_pad % 2),
                          (x_pad // 2, x_pad // 2 + x_pad % 2),
                          (0, 0)),
                      mode='constant')

    if x_pad < 0:
        x_crop_l, x_crop_r = abs(x_pad) // 2, abs(x_pad) // 2 + abs(x_pad) % 2
        a = a[:, x_crop_l:-x_crop_r]
    elif x_pad > 0:
        a = np.pad(a, ((0, 0), (x_pad // 2, x_pad // 2 + x_pad % 2), (0, 0)), mode='constant')

    if y_pad < 0:
        y_crop_t, y_crop_b = abs(y_pad) // 2, abs(y_pad) // 2 + abs(y_pad) % 2
        a = a[y_crop_t:-y_crop_b]
    elif y_pad > 0:
        a = np.pad(a, ((y_pad // 2, y_pad // 2 + y_pad % 2), (0, 0), (0, 0)), mode='constant')

    return a


def pad_to_scan_size(target, scan):
    target = {k: to_shape(v, scan.shape) for k, v in target.items()}
    return target


def compute_diff(pred, gt):
    trab_diff = np.abs(pred["trab_mask"] - gt["trab_mask"])
    cort_diff = np.abs(pred["cort_mask"] - gt["cort_mask"])
    return np.clip(trab_diff + cort_diff, 0, 1)


def keep_largest_cc(img):
    labels, num_labels = sklabel(img, background=0, return_num=True)
    max_lbl, max_count = 1, 0
    for lbl in range(1, num_labels):
        count = np.sum(labels == lbl)
        if count > max_count:
            max_lbl = lbl
            max_count = count
    img[labels != max_lbl] = 0
    return img


def sample_padder(sample, img_size_mult, pad_mode_='edge', augmentation=False):
    # when a sample is passed through this transformation, the output
    # should be of images and masks that are all of the same size, aligned,
    # and who have x and y dim lengths the same and a multiple of the integer
    # provided at object initialization.
    volumes = ['image', 'cort_mask', 'trab_mask']
    volumes = list(set(volumes).intersection(sample.keys()))

    # Step 1: Figure out the upper and lower x,y bounds for the union of
    # the image and masks

    lower_x = min([sample[f"{v}_position"][0] for v in volumes])
    upper_x = max([sample[f"{v}_position"][0] + sample[v].shape[0] for v in volumes])

    lower_y = min([sample[f"{v}_position"][1] for v in volumes])
    upper_y = max([sample[f"{v}_position"][1] + sample[v].shape[1] for v in volumes])

    # Step 2: Calculate the new size for the image, which will be the larger
    # of the width and height, then rounded up to the next highest multiple

    width = upper_x - lower_x
    height = upper_y - lower_y

    padded_size = img_size_mult * math.ceil(max(width, height) / img_size_mult)

    # Step 3: Calculate the new upper/lower bounds in x and y

    lower_x_padded = lower_x - (padded_size - width) // 2
    upper_x_padded = lower_x_padded + padded_size

    lower_y_padded = lower_y - (padded_size - height) // 2
    upper_y_padded = lower_y_padded + padded_size

    # Step 4: Pad the image and masks to the new bounds, and adjust the
    # position entries in the sample dict

    for v in volumes:

        if v == 'image':
            pad_mode = pad_mode_
        else:
            pad_mode = 'constant'

        sample[v] = np.pad(sample[v],
            (
                (
                    sample[f"{v}_position"][0] - lower_x_padded,
                    upper_x_padded - (sample[f"{v}_position"][0] + sample[v].shape[0])
                ),
                (
                    sample[f"{v}_position"][1] - lower_y_padded,
                    upper_y_padded - (sample[f"{v}_position"][1] + sample[v].shape[1])
                ),
                (0, 0)
            ),
            mode=pad_mode
        )

        sample[f'{v}_position'][0] = lower_x_padded
        sample[f'{v}_position'][1] = lower_y_padded
    return sample


def shift_masks(pred_sample, gt_sample):
    pred_padded = sample_padder(pred_sample, 8)
    gt_sample["image"] = pred_padded["image"]
    gt_sample["image_position"] = pred_padded["image_position"]
    gt_padded = sample_padder(gt_sample, 8)
    pred = dict(cort_mask=pred_padded["cort_mask"], trab_mask=pred_padded["trab_mask"])
    gt = dict(cort_mask=gt_padded["cort_mask"], trab_mask=gt_padded["trab_mask"])
    scan = pred_sample["image"]
    return scan, pred, gt


# def shift_gt(pred, gt):
#     pred_cort, pred_trab = pred["cort_mask"], pred["trab_mask"]
#     gt_cort, gt_trab = gt["cort_mask"], gt["trab_mask"]
#     pred_cort_slc, pred_trab_slc = pred_cort[..., 168 // 2], pred_trab[..., 168 // 2]
#     gt_cort_slc, gt_trab_slc = gt_cort[..., 168 // 2], gt_trab[..., 168 // 2]
#
#     # Combine cortical and trabecular masks and get the indices of all nonzero entries to compute the center of mass
#     pred_slc = np.clip(pred_cort_slc + pred_trab_slc, 0, 1)
#     pred_slc = keep_largest_cc(pred_slc)
#     pred_nz_rows, pred_nz_cols = pred_slc.nonzero()
#     pred_avg_row, pred_avg_col = pred_nz_rows.mean(), pred_nz_cols.mean()
#     gt_slc = np.clip(gt_cort_slc + gt_trab_slc, 0, 1)
#     gt_nz_rows, gt_nz_cols = gt_slc.nonzero()
#     gt_avg_row, gt_avg_col = gt_nz_rows.mean(), gt_nz_cols.mean()
#     # Compute the distance col- and row-wise between the centers of masses
#     row_shift = np.round(pred_avg_row - gt_avg_row).astype(int)
#     col_shift = np.round(pred_avg_col - gt_avg_col).astype(int)
#     # Shift the ground truth
#     gt_cort = np.roll(gt_cort, (row_shift, col_shift), axis=(0, 1))
#     gt_trab = np.roll(gt_trab, (row_shift, col_shift), axis=(0, 1))
#
#     return dict(trab_mask=pred_trab, cort_mask=pred_cort), dict(trab_mask=gt_trab, cort_mask=gt_cort)


def plot_img(scan, slice_idx, output_path, scan_id, cort=None, trab=None, show=False, save=False, mask_type=""):
    slice_ = scan[..., slice_idx]
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.imshow(slice_, cmap='gray')
    if cort is not None:
        cort_ = cort[..., slice_idx]
        ax.imshow(cort_, cmap='Greens', alpha=0.6 * cort_)
    if trab is not None:
        trab_ = trab[..., slice_idx]
        ax.imshow(trab_, cmap='Blues', alpha=0.6 * trab_)
    ax.set_axis_off()
    if save and cort is None and trab is None:
        plt.savefig(Path(output_path, f"{scan_id}_scan_{slice_idx}.png"), bbox_inches="tight")
    elif save and cort is not None and trab is not None:
        plt.savefig(Path(output_path, f"{scan_id}_scan_with_{mask_type}_masks_{slice_idx}.png"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(plt.gcf())


def plot_slice_with_overlay(scan, pred, gt, slice_idx, scan_id, output_path, show=False, save=False):
    pred = {k: v[..., slice_idx] for k, v in pred.items()}
    gt = {k: v[..., slice_idx] for k, v in gt.items()}
    slice_ = scan[..., slice_idx]

    fig, ax = plt.subplots(2, 2, figsize=(20, 12))

    # Plot original image
    ax[0, 0].imshow(slice_, cmap='gray')
    ax[0, 0].set_title("Original image")
    ax[0, 0].set_axis_off()

    # Plot img with gt masks
    ax[1, 0].imshow(slice_, cmap='gray')
    ax[1, 0].imshow(gt["trab_mask"], cmap='Blues', alpha=0.7 * gt["trab_mask"])
    ax[1, 0].imshow(gt["cort_mask"], cmap='Greens', alpha=0.7 * gt["cort_mask"])
    ax[1, 0].set_title("gt")
    ax[1, 0].set_axis_off()

    # Plot img with pred masks
    ax[1, 1].imshow(slice_, cmap='gray')
    ax[1, 1].imshow(pred["trab_mask"], cmap='Blues', alpha=0.7 * pred["trab_mask"])
    ax[1, 1].imshow(pred["cort_mask"], cmap='Greens', alpha=0.7 * pred["cort_mask"])
    ax[1, 1].set_title("pp")
    ax[1, 1].set_axis_off()

    # Plot img with overlay
    ax[0, 1].imshow(slice_, cmap='gray')
    ax[0, 1].imshow(pred["trab_mask"], cmap='Blues', alpha=0.7 * pred["trab_mask"])
    ax[0, 1].imshow(pred["cort_mask"], cmap='Greens', alpha=0.7 * pred["cort_mask"])
    diff = compute_diff(pred, gt)
    ax[0, 1].imshow(diff, cmap='Reds', alpha=0.7 * diff)
    ax[0, 1].set_title("Overlay of gt and pp mask (red pixels are differences)")
    ax[0, 1].set_axis_off()

    diff_cort_pixels = np.sum(np.abs(gt["cort_mask"] - pred["cort_mask"]))
    diff_trab_pixels = np.sum(np.abs(gt["trab_mask"] - pred["trab_mask"]))
    diff_all_pixels = diff_cort_pixels + diff_trab_pixels

    plt.suptitle(f"gt & pp masks (with overlay) (slice index={slice_idx} & scan id={scan_id})\n"
                 f"#different pixels: cort={diff_cort_pixels}, trab={diff_trab_pixels}, all={diff_all_pixels}\n"
                 "(blue = trabecular, green = cortical, red = difference)")
    if save:
        plt.savefig(Path(output_path, f"{scan_id}_all_overlays_{slice_idx}.png"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(plt.gcf())


def plot_overlay(scan, pred, gt, slice_idx, output_path, scan_id, show=False, save=False):
    pred = {k: v[..., slice_idx] for k, v in pred.items()}
    gt = {k: v[..., slice_idx] for k, v in gt.items()}
    slice_ = scan[..., slice_idx]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(slice_, cmap='gray')
    ax.imshow(pred["trab_mask"], cmap='Blues', alpha=0.7 * pred["trab_mask"])
    ax.imshow(pred["cort_mask"], cmap='Greens', alpha=0.7 * pred["cort_mask"])
    diff = compute_diff(pred, gt)
    ax.imshow(diff, cmap='Reds', alpha=0.7 * diff)
    # ax.set_title("Overlay of predicted and reference mask (red pixels are differences)")
    ax.set_axis_off()
    if save:
        plt.savefig(Path(output_path, f"{scan_id}_overlay_{slice_idx}.png"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(plt.gcf())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_gt_dir")
    parser.add_argument("--path_to_pred_dir")
    parser.add_argument("--output_path")
    parser.add_argument("--path_to_gt_masks_dir", default="")
    parser.add_argument("--show", default="")
    parser.add_argument("--save", default="")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--all_slices", default="")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    pred_mask_dir = Path(args.path_to_pred_dir)
    gt_dir = Path(args.path_to_gt_dir)
    scan_ids = [f.stem[:f.stem.index("_")] for f in pred_mask_dir.glob("*_TRAB_MASK.AIM")]
    gt_mask_dir = Path(args.path_to_gt_masks_dir) if args.path_to_gt_masks_dir else False
    if args.all_slices:
        try:
            slice_idcs = [int(args.all_slices)]
        except:
            slice_idcs = list(range(0, 168, 5))
    else:
        slice_idcs = [60]

    i = 1
    # scan_id_to_slice_idx = {"c0002321": [94, 95, 96, 97, 98, 104, 105, 106, 107], "c0002261": [0], "c0002304": }
    scan_id_to_slice_idx = None
    scan_ids = ["c0002261", "c0002304", "c0002321"]  #list(scan_id_to_slice_idx.keys())

    for scan_id in scan_ids:
        print(f"{i}/{len(scan_ids)} - {scan_id}", flush=True, end=" ")
        slice_idcs = scan_id_to_slice_idx.get(scan_id, -1) if scan_id_to_slice_idx else slice_idcs
        scan, scan_md = load_scan(gt_dir, scan_id)
        scan = scan[:-1, :-1]
        pred, pred_md = load_masks(pred_mask_dir, scan_id)#, suffix=args.suffix)
        if gt_mask_dir:
            gt, gt_md = load_masks(gt_mask_dir, scan_id, suffix=args.suffix)
        else:
            gt, gt_md = {k: pred[k].copy() for k in pred.keys()}, pred_md.copy()

        # Linear registration to ensure pred and gt are in the same location
        pred_sample = {"cort_mask_position": pred_md["cort_md"]["position"],
                       "trab_mask_position": pred_md["trab_md"]["position"],
                       "image_position": scan_md["position"],
                       "image": scan, **pred}
        gt_sample = {"cort_mask_position": gt_md["cort_md"]["position"],
                     "trab_mask_position": gt_md["trab_md"]["position"],
                     "image_position": scan_md["position"],
                     "image": scan, **gt}
        scan, pred, gt = shift_masks(pred_sample, gt_sample)

        for slice_idx in slice_idcs:
            print(slice_idx, end=" ")
            # plot_img(scan, slice_idx, args.output_path, scan_id, show=args.show, save=args.save)
            plot_img(scan, slice_idx, args.output_path, scan_id, cort=gt['cort_mask'], trab=gt['trab_mask'],
                     show=args.show, save=args.save, mask_type='gt')
            plot_img(scan, slice_idx, args.output_path, scan_id, cort=pred['cort_mask'], trab=pred['trab_mask'],
                     show=args.show, save=args.save, mask_type='pred')
            plot_overlay(scan, pred, gt, slice_idx, args.output_path, scan_id, show=args.show, save=args.save)
            plot_slice_with_overlay(scan, pred, gt, slice_idx, scan_id, args.output_path, args.show, args.save)
        print()
        i += 1


if __name__ == "__main__":
    main()
    # data_dir = "/Users/fbergh/Documents/viecuri/segmentatie/data.nosync/xtremect1/HUG/"
    # scan_id = "C0004417"
    # scan, scan_md = load_scan(data_dir, scan_id)
    # scan = scan[:-1, :-1]
    # auto_masks, auto_md = load_masks(data_dir, scan_id)
    # sauto_mask, sauto_md = load_aim(data_dir + f"{scan_id}_MASK.AIM")
    # sauto_md = dict(cort_md=deepcopy(sauto_md), trab_md=deepcopy(sauto_md))
    # sauto_mask //= 127
    # sauto_masks = dict(cort_mask=np.copy(sauto_mask), trab_mask=np.copy(sauto_mask))
    #
    # sauto_sample = {"cort_mask_position": sauto_md["cort_md"]["position"],
    #                "trab_mask_position": sauto_md["trab_md"]["position"],
    #                "image_position": scan_md["position"],
    #                "image": scan, **sauto_masks}
    # auto_sample = {"cort_mask_position": auto_md["cort_md"]["position"],
    #              "trab_mask_position": auto_md["trab_md"]["position"],
    #              "image_position": scan_md["position"],
    #              "image": scan, **auto_masks}
    # scan, sauto_masks, auto_masks = shift_masks(sauto_sample, auto_sample)
    #
    # os.makedirs(data_dir + "overlays_auto", exist_ok=True)
    # os.makedirs(data_dir + "overlays_sauto", exist_ok=True)
    # plot_img(scan, scan.shape[-1] // 2, data_dir + "figures", scan_id, save=True)
    # plot_overlay(scan, auto_masks, auto_masks, scan.shape[-1] // 2, data_dir + "overlays_auto", scan_id, False, True)
    # plot_overlay(scan, sauto_masks, sauto_masks, scan.shape[-1] // 2, data_dir + "overlays_sauto", scan_id, False, True)

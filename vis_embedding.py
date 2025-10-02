import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.widgets import RangeSlider, Slider
from tqdm import tqdm

from compare import to_shape
from test_grounds import load_aim, rescale
from utils.error_metrics import create_convert_embeddings_to_predictions


def roundup(x):
    if x > 0:
        return int(math.ceil(x / 100.0)) * 100
    else:
        return int(math.floor(x / 100.0)) * 100


def threshold_masks(peri_emb_, endo_emb_, thresh):
    cort_mask = (peri_emb_ < thresh) * (endo_emb_ > thresh)
    trab_mask = endo_emb_ < thresh
    return cort_mask, trab_mask


def plot_interactive_mask(scan, endo_emb, peri_emb, thresh=0):
    f, ax = plt.subplots(1, 1)
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    slider = Slider(slider_ax, "Threshold", valmin=-2.5, valmax=2.5, valinit=0)
    cort_mask, trab_mask = threshold_masks(peri_emb, endo_emb, thresh)

    def plot_overlay(cort_mask_, trab_mask_):
        ax.clear()
        ax.set_axis_off()
        ax.imshow(scan, cmap="gray")
        ax.imshow(trab_mask_, cmap='Blues', alpha=0.7 * trab_mask_, vmin=0, vmax=1)
        ax.imshow(cort_mask_, cmap='Greens', alpha=0.7 * cort_mask_, vmin=0, vmax=1)

    def update(threshold):
        cort_mask_, trab_mask_ = threshold_masks(peri_emb, endo_emb, threshold)
        plot_overlay(cort_mask_, trab_mask_)
        f.canvas.draw_idle()

    plot_overlay(cort_mask, trab_mask)
    slider.on_changed(update)
    plt.show()


def plot_certainty_mask(scan, certainty_mask, mask_type):
    cmap = "Greens" if mask_type == "cort" else "Blues" if mask_type == "trab" else "Reds"
    plt.figure()
    plt.imshow(scan, cmap='gray')
    im = plt.imshow(certainty_mask, cmap=cmap, alpha=(certainty_mask > 0).astype(float))
    plt.axis('off')
    plt.title(f"{mask_type} certainty map")
    plt.colorbar(im)
    plt.show()
    plt.close()


def plot_one_certainty_mask(scan, certainty_map_dict):
    plt.figure()
    plt.imshow(scan, cmap='gray')
    for mask_type, certainty_map in certainty_map_dict.items():
        cmap = "Greens" if mask_type == "cort" else "Blues" if mask_type == "trab" else "Reds"
        im = plt.imshow(certainty_map, cmap=cmap, alpha=0.7 * certainty_map)
    plt.axis('off')
    plt.show()
    plt.close()


def compute_certainty(endo_emb, peri_emb, step_size=1., min_val=None, max_val=None):
    if min_val is None:
        min_val = min(endo_emb.min(), peri_emb.min())
    if max_val is None:
        max_val = min(endo_emb.max(), peri_emb.max())
    n_vals = max_val - min_val
    bg_cumsum = np.zeros(endo_emb.shape, dtype=int)
    cort_cumsum = np.zeros_like(bg_cumsum)
    trab_cumsum = np.zeros_like(bg_cumsum)
    n_steps = round(n_vals / step_size)
    for thresh in tqdm(np.linspace(min_val, max_val, endpoint=True, num=n_steps)):
        cort_mask, trab_mask = threshold_masks(peri_emb, endo_emb, thresh)
        bg_cumsum += (~cort_mask & ~trab_mask)
        cort_cumsum += cort_mask
        trab_cumsum += trab_mask
    return dict(background=bg_cumsum / n_steps, cort=cort_cumsum / n_steps, trab=trab_cumsum / n_steps)


def main():
    # base_path = "/home/fvdenbergh/segmentation"
    base_path = "/Users/fbergh/Documents/viecuri/segmentatie"
    dataset = "fxmovie"
    scan_id = "c0001139"
    # slices = [167] #np.linspace(0, 167, num=17, dtype=int)
    if dataset == "fxmovie":
        # scan, m, b = load_aim(f"{base_path}/data/{dataset}/batch1/{scan_id}_crop.aim")
        # endo_emb = np.load(f"{base_path}/data/{dataset}/batch1/scon_embedding/{scan_id}_crop_endo_embedding.npy")
        # peri_emb = np.load(f"{base_path}/data/{dataset}/batch1/scon_embedding/{scan_id}_crop_peri_embedding.npy")
        scan, m, b = load_aim(f"{base_path}/data.nosync/{dataset}/batch1/{scan_id}_crop.aim")
        endo_emb = np.load(f"{base_path}/results.nosync/{dataset}/batch1/scon/embedding/{scan_id}_crop_endo_embedding.npy")
        peri_emb = np.load(f"{base_path}/results.nosync/{dataset}/batch1/scon/embedding/{scan_id}_crop_peri_embedding.npy")
    elif dataset == "skedoi":
        scan, m, b = load_aim(f"{base_path}/data/{dataset}/{scan_id}_crop.aim")
        endo_emb = np.load(f"{base_path}/data/{dataset}/embedding/{scan_id}_crop_endo_embedding.npy")
        peri_emb = np.load(f"{base_path}/data/{dataset}/embedding/{scan_id}_crop_peri_embedding.npy")
    else:
        scan, m, b = None, 0, 0
        endo_emb, peri_emb = 0, 0
    scan = to_shape(rescale(scan, m, b), endo_emb.shape)

    slice_idx = 167

    plt.figure()
    plt.imshow(scan[..., slice_idx], cmap='gray')
    plt.axis('off')
    plt.show()

    plt.figure()
    im = plt.imshow(np.clip(endo_emb[..., slice_idx], -200, 200), cmap='PiYG')
    plt.colorbar(im)
    plt.show()
    plt.close()
    plt.figure()
    im = plt.imshow(np.clip(peri_emb[..., slice_idx], -200, 200), cmap='PiYG')
    plt.colorbar(im)
    plt.show()
    plt.close()

    plot_interactive_mask(scan[..., 167], endo_emb[..., 167], peri_emb[..., 167])

    # to_certainty_map = create_convert_embeddings_to_predictions(0.1)
    # emb = torch.Tensor(np.stack([peri_emb, endo_emb], axis=1))
    # cmap = torch.exp(to_certainty_map(emb)).numpy()
    # certainty_map_dict = dict(background=cmap[:, 2, :, slice_idx],
    #                           cort=cmap[:, 0, :, slice_idx],
    #                           trab=cmap[:, 1, :, slice_idx])
    #
    # # certainty_map_dict = compute_certainty(endo_emb[..., slice_idx], peri_emb[..., slice_idx], step_size=0.1)#, min_val=-100, max_val=100)
    #
    # for mask_type, certainty_map in certainty_map_dict.items():
    #     plot_certainty_mask(scan[..., slice_idx], certainty_map, mask_type)
    # plot_one_certainty_mask(scan[..., slice_idx], certainty_map_dict)


if __name__ == "__main__":
    main()

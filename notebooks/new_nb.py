#%%
import numpy as np
from ipywidgets import interact
from matplotlib import pyplot as plt

from test import rescale, load_aim

dataset = "fxmovie"
scan_id = "c0001139"
slices = [167] #np.linspace(0, 167, num=17, dtype=int)
base_path = "/home/fvdenbergh/segmentation/data"




if dataset == "fxmovie":
    scan, m, b = load_aim(f"{base_path}/{dataset}/batch1/{scan_id}_crop.aim")
    endo_emb = np.load(f"{base_path}/{dataset}/batch1/embedding/{scan_id}_crop_endo_embedding.npy")
    peri_emb = np.load(f"{base_path}/{dataset}/batch1/embedding/{scan_id}_crop_peri_embedding.npy")
elif dataset == "skedoi":
    scan, m, b = load_aim(f"{base_path}/{dataset}/{scan_id}_crop.aim")
    endo_emb = np.load(f"{base_path}/{dataset}/embedding/{scan_id}_crop_endo_embedding.npy")
    peri_emb = np.load(f"{base_path}/{dataset}/embedding/{scan_id}_crop_peri_embedding.npy")
else:
    scan, m, b = None, 0, 0
    endo_emb, peri_emb = 0, 0
scan = rescale(scan, m, b)

endo_emb_norm = (endo_emb - endo_emb.min()) / (endo_emb.max() - endo_emb.min())
peri_emb_norm = (peri_emb - peri_emb.min()) / (peri_emb.max() - peri_emb.min())




def threshold_masks(peri_emb_, endo_emb_, peri_thresh, endo_thresh):
    cort_mask = (peri_emb_ < peri_thresh) * (endo_emb_ > endo_thresh)
    trab_mask = endo_emb_ < endo_thresh
    return cort_mask, trab_mask

# cort_mask = (phi_peri < 0) * (phi_endo > 0)
# trab_mask = phi_endo < 0
%matplotlib inline

def show_embeddings(peri_thresh=0, endo_thresh=0):
    for slice_idx in slices:
        f, ax = plt.subplots(2, 3, figsize=(12, 4))
        ax[0, 0].imshow(scan[..., slice_idx], cmap="gray")
        ax[0, 0].set_title(f"Original scan (slice {slice_idx + 1})")
        ax[0, 0].axis('off')
        ax[0, 1].imshow(endo_emb_norm[..., slice_idx], cmap="RdYlBu")
        ax[0, 1].set_title(f"Endosteal embedding (slice {slice_idx + 1})\n(neg (trab)=red, pos (cort)=blu)")
        ax[0, 1].axis('off')
        ax[0, 2].imshow(peri_emb_norm[..., slice_idx], cmap="RdYlBu")
        ax[0, 2].set_title(f"Periosteal embedding (slice {slice_idx + 1})\n(neg (cort)=red, pos (back)=blu)")
        ax[0, 2].axis('off')

        cort_mask, trab_mask = threshold_masks(peri_emb, endo_emb, peri_thresh, endo_thresh)

        ax[1, 0].imshow(scan[..., slice_idx], cmap="gray")
        ax[1, 0].imshow(trab_mask, cmap='Blues', alpha=0.7 * trab_mask)
        ax[1, 0].imshow(cort_mask, cmap='Greens', alpha=0.7 * cort_mask)
        ax[1, 0].set_title(f"Overlay (slice {slice_idx + 1})")
        ax[1, 0].axis('off')
        ax[1, 1].imshow(trab_mask, cmap="gray")
        ax[1, 1].set_title(f"Trab mask (slice {slice_idx + 1})")
        ax[1, 1].axis('off')
        ax[1, 2].imshow(cort_mask, cmap="gray")
        ax[1, 2].set_title(f"Cort mask (slice {slice_idx + 1})")
        ax[1, 2].axis('off')
        plt.show()


interact(show_embeddings, peri_thresh=(-5., 5., 0.01), cort_thresh=(-5., 5., 0.01))
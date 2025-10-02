"""
Written by Nathan Neeteson
Utilities for quantitatively comparing predicted and reference segmentations.
Loosely adapted from: https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/34_Segmentation_Evaluation.ipynb
"""

import SimpleITK as sitk
import numpy as np


def binarize_numpy_array(arr):
    return (np.abs(arr) > 0).astype(np.int)


# take in an ITK image mask and give the dist map and surface images
def get_distance_map_and_surface(mask, do_abs=True):
    dist_map = sitk.SignedMaurerDistanceMap(
            mask, squaredDistance=False, useImageSpacing=True
        )
    if do_abs:
        dist_map = sitk.Abs(dist_map)
    surface = sitk.LabelContour(mask, fullyConnected=False)
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(surface)
    surface_num_pix = int(stats_filter.GetSum())
    return dist_map, surface, surface_num_pix


def get_surface_to_surface_distances_list(surf2surf_dist_map, surface_num_pix):
    surf2surf_dist_array = sitk.GetArrayFromImage(surf2surf_dist_map).flatten()
    surf2surf_dist_list = list(surf2surf_dist_array[surf2surf_dist_array != 0])
    num_nonzero_pix = len(surf2surf_dist_list)
    if num_nonzero_pix < surface_num_pix:
        zeros_list = list(np.zeros(surface_num_pix - num_nonzero_pix))
        surf2surf_distance_list = surf2surf_dist_list + zeros_list

    return surf2surf_dist_list


# dice similarity score
def calculate_dice_and_jaccard(ref, seg):
    ref, seg = ref > 0, seg > 0
    ref, seg = ref.flatten(), seg.flatten()
    dice = 2 * (ref & seg).sum() / (ref.sum() + seg.sum())
    jaccard = (ref & seg).sum() / (ref | seg).sum()
    return dice, jaccard

def _plot(img, slc=50):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img[..., slc])
    plt.show()


# hausdorff distance
def calculate_surface_distance_measures(ref, seg, spacing, vis=True, scan=None):
    # sitk.ProcessObject_SetGlobalDefaultThreader('PLATFORM')
    # ref_np = ref // 127
    # seg_np = seg // 127
    # binarize reference and segmentation
    ref, seg = binarize_numpy_array(ref), binarize_numpy_array(seg)
    # print("binarized")

    # convert numpy binary matrices into binary SITK images
    ref = sitk.GetImageFromArray(ref)
    seg = sitk.GetImageFromArray(seg)
    ref.SetSpacing(spacing)
    seg.SetSpacing(spacing)
    # print("converted into binary sitk")

    # calculate the surface and distance maps for reference and segmentation
    ref_dist_map, ref_surface, ref_surface_num_pix = get_distance_map_and_surface(ref)
    seg_dist_map, seg_surface, seg_surface_num_pix = get_distance_map_and_surface(seg)
    # ref_dist_map_np = sitk.GetArrayFromImage(ref_dist_map)
    # _plot(ref_dist_map_np, 44)
    # seg_surface_np = sitk.GetArrayFromImage(seg_surface)
    # _plot(seg_surface_np, 44)
    # seg_dist_map_np = sitk.GetArrayFromImage(seg_dist_map)
    # _plot(seg_dist_map_np, 44)
    # ref_surface_np = sitk.GetArrayFromImage(ref_surface)
    # _plot(ref_surface_np, 44)
    # print("calculated surface")

    # get the symmetric distances by multiplying the reference distance map by
    # the segmentation surface and vice versa
    seg2ref_dist_map = ref_dist_map * sitk.Cast(seg_surface, sitk.sitkFloat32)
    ref2seg_dist_map = seg_dist_map * sitk.Cast(ref_surface, sitk.sitkFloat32)
    # print("symmetric distances")

    # seg2ref_dist_map_np = sitk.GetArrayFromImage(seg2ref_dist_map)
    # ref2seg_dist_map_np = sitk.GetArrayFromImage(ref2seg_dist_map)
    # t = ref2seg_dist_map_np if ref2seg_dist_map_np.max() > seg2ref_dist_map_np.max() else seg2ref_dist_map_np
    # t *= 1_000
    # t_max = t.max()
    # print(t_max)
    # print('took max')

    # import matplotlib.pyplot as plt
    # r, c, large_err_slices = (t > t_max * 0.9).nonzero()
    # test_img = np.zeros(ref_np.shape[:2])
    # for r, c in zip(r, c):
    #     test_img[r, c] = 1
    # print('found nonzero')
    # print(large_err_slices)
    # large_err_slices = np.unique(large_err_slices)
    # print(large_err_slices)
    # print("took unique")
    # if vis: #and len(large_err_slices) <= 5:
    #     for slc in large_err_slices:
    #         plt.figure()
    #         plt.axis('off')
    #         if scan is not None:
    #             plt.imshow(scan[..., slc], cmap="gray")
    #         plt.imshow(ref_np[..., slc], cmap='Blues', alpha=0.5 * ref_np[..., slc], label="ref")
    #         plt.imshow(seg_np[..., slc], cmap='Reds', alpha=0.5 * seg_np[..., slc], label="pred")
    #         # plt.imshow(test_img, cmap="Greens", alpha=1*test_img, label="error region")
    #         # plt.imshow(np.abs(seg_np[..., slc] - ref_np[..., slc]), cmap='Reds', alpha=0.7 * np.abs(seg_np[..., slc] - ref_np[..., slc]))
    #         plt.title(slc)
    #         plt.show()
    #         plt.close(plt.gcf())
    # return large_err_slices.tolist()

    # get lists of the distances (including overlap)
    seg2ref_dist_list = \
        get_surface_to_surface_distances_list(seg2ref_dist_map, seg_surface_num_pix)
    ref2seg_dist_list = \
        get_surface_to_surface_distances_list(ref2seg_dist_map, ref_surface_num_pix)
    all_dist_list = seg2ref_dist_list + ref2seg_dist_list

    # calculate max, median, mean, std of symmetric surface distances
    ssd_measures = {}
    ssd_measures['max'] = np.max(all_dist_list)
    ssd_measures['median'] = np.median(all_dist_list)
    ssd_measures['mean'] = np.mean(all_dist_list)
    ssd_measures['std'] = np.std(all_dist_list)

    return ssd_measures

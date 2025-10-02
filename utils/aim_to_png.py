import argparse
import os
import warnings
from pathlib import Path

import vtkbone
import matplotlib.pyplot as plt
from tqdm import tqdm
from vtkmodules.util.numpy_support import vtk_to_numpy
# from bonelab.util.aim_calibration_header import get_aim_density_equation


def aim_to_png(path_to_aim, output_dir, filename="", save=False, plot=False, all_slices=None):
    path_to_aim = str(path_to_aim)
    if "9018" in path_to_aim:
        warnings.warn("Scan ID 9018 doesn't work for some reason")
        return
    if filename == "":
        filename = path_to_aim.split("/")[-1].split(".")[0]
    reader = vtkbone.vtkboneAIMReader()
    reader.DataOnCellsOff()
    reader.SetFileName(path_to_aim)
    reader.Update()

    vtk_data = reader.GetOutput().GetPointData().GetScalars()
    data = vtk_to_numpy(vtk_data).reshape(reader.GetOutput().GetDimensions(), order='F')

    # m, b = get_aim_density_equation(reader.GetProcessingLog())
    # data = m * data + b
    if all_slices:
        try:
            slice_idcs = [int(all_slices)]
        except:
            slice_idcs = list(range(0, 168, 1))
    else:
        slice_idcs = [60]

    for slice_idx in slice_idcs:
        plt.figure()
        plt.imshow(data[..., slice_idx], cmap='gray')
        plt.axis('off')
        if save:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{filename}_{slice_idx}.png"), bbox_inches='tight')
        if plot:
            plt.show()
        plt.close(plt.gcf())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_aim")
    parser.add_argument("--output_dir")
    parser.add_argument("--filename", default="")
    parser.add_argument("--save", default="")
    parser.add_argument("--plot", default="")
    parser.add_argument("--all_slices", default="")
    args = parser.parse_args()
    path_to_aim = Path(args.path_to_aim)
    if path_to_aim.is_dir():
        print(f'Processing dir: {path_to_aim}')
        args_dict = args.__dict__
        del args_dict['path_to_aim']
        for p in tqdm(list(path_to_aim.glob("*.AIM"))):
            aim_to_png(p, **args_dict)
        for p in tqdm(list(path_to_aim.glob("*.aim"))):
            aim_to_png(p, **args_dict)
    else:
        print(f"Processing file: {path_to_aim}")
        aim_to_png(**args.__dict__)


if __name__ == "__main__":
    main()

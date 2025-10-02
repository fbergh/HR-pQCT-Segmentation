"""
Written by Nathan Neeteson.
Use a UNet model to infer a trabecular and cortical mask for an image.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.SingleImageDataset import SingleImageDataset


def infer(
        args,
        model,
        device,
        image,
        image_dataset_kwargs,
        image_dataloader_kwargs
):
    model.eval()

    phi_peri = np.zeros((
        image['image'].shape[2],
        image['image'].shape[3],
        image['image'].shape[4]
    ))
    phi_endo = np.zeros((
        image['image'].shape[2],
        image['image'].shape[3],
        image['image'].shape[4]
    ))

    image_dataset = SingleImageDataset(**image_dataset_kwargs)
    image_dataset.setup_new_image(image)

    image_dataloader = DataLoader(image_dataset, **image_dataloader_kwargs)

    for idx, image_slice in enumerate(image_dataloader):
        # i = image_slice['image']
        # print('input data', torch.any(i == 0), torch.all(i == 0), torch.all(i != 0))
        # itd = i.to(device)
        # print('input data', torch.any(itd == 0), torch.all(itd == 0), torch.all(itd != 0))
        data = image_slice['image'].to(device)
        # print('input data', torch.any(data == 0), torch.all(data == 0), torch.all(data != 0))

        with torch.no_grad():
            embeddings = model(data).squeeze(0).cpu().detach().numpy()

        phi_peri[:, :, idx] = embeddings[0, :, :]
        phi_endo[:, :, idx] = embeddings[1, :, :]

        # print('==========')

    return phi_peri, phi_endo

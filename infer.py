"""
Written by Nathan Neeteson
Segment images using a trained U-Net and optionally, compare to reference segmentations.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from dataset.HRpQCTAIMDataset import HRpQCTAIMDataset
from dataset.SamplePadder import SamplePadder
from dataset.SampleStandardizer import SampleStandardizer
from dataset.SampleToTensors import SampleToTensors
from models.UNet import UNet
from traintest.infer import infer
from utils.image_export import save_mask_as_AIM, save_numpy_array_as_image
from utils.logger import Logger
from utils.postprocessing import postprocess_masks_iterative
from utils.segmentation_evaluation import (
    calculate_dice_and_jaccard, calculate_surface_distance_measures
)

# PROBLEM_FILES = ["c0002261_crop", "c0001139_crop"]
# Problem files for the acon
# PROBLEM_FILES = ["c0001139_crop", "c0002174_crop"]
PROBLEM_FILES = []


def check_odd_integer(value):
    ivalue = int(value)
    if not (ivalue % 2):
        raise argparse.ArgumentTypeError(f"{ivalue} is not a valid odd integer")
    return ivalue


def create_parser():
    parser = argparse.ArgumentParser(
        description='HRpQCT Segmentation 2D UNet Inference Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-dir', type=str, default='./data/test/', metavar='STR',
        help='path of directory containing images to do inference on'
    )

    parser.add_argument(
        '--predictions-suffix', type=str, default='scon', metavar='STR',
        help='path of directory containing images to do inference on'
    )

    parser.add_argument(
        '--data-pattern', type=str, default='*.AIM', metavar='STR',
        help='glob-like data pattern for images to train/validate on'
    )

    parser.add_argument(
        '--trained-model', type=str, default='./trained_models/', metavar='STR',
        help='path of trained model'
    )

    parser.add_argument(
        '--model-filters', type=int, nargs='+', default=[32, 64, 128, 256], metavar='N',
        help='sequence of filters in U-Net layers'
    )

    parser.add_argument(
        '--channels-per-group', type=int, default=16, metavar='N',
        help='channels per group in GroupNorm'
    )

    parser.add_argument(
        '--dropout', type=float, default=0.1, metavar='D',
        help='dropout probability'
    )

    parser.add_argument(
        '--min-density', type=float, default=-400, metavar='D',
        help='minimum physiologically relevant density in the image [mg HA/ccm]'
    )

    parser.add_argument(
        '--max-density', type=float, default=1400, metavar='D',
        help='maximum physiologically relevant density in the image [mg HA/ccm]'
    )

    parser.add_argument(
        '--input-channels', type=check_odd_integer, default=5, metavar='N',
        help='number slices to use as input for each slice prediction, must be odd'
    )

    parser.add_argument(
        '--output-channels', type=int, default=2, metavar='N',
        help='number of channels in the output data'
    )

    parser.add_argument(
        '--cuda', action='store_true', default=False,
        help='enable cuda processing'
    )

    parser.add_argument(
        '--load-ref-masks', action='store_true', default=False,
        help='load reference masks'
    )

    parser.add_argument(
        '--evaluate', action='store_true', default=False,
        help='evaluate quality of predictions (requires reference masks)'
    )

    parser.add_argument(
        '--only-evaluate', action='store_true', default=False,
        help='only evaluate quality of predictions, do not predict (requires reference and prediction masks)'
    )

    parser.add_argument(
        '--prediction-path', default=".", help="Path to find predictions when only evaluating"
    )

    parser.add_argument(
        '--spacing', type=float, default=61e-6, metavar='S',
        help='isometric voxel width [m]'
    )

    parser.add_argument(
        '--visualize', action='store_true', default=False,
        help='plot the masks and also save a bunch of images'
    )

    parser.add_argument(
        '--write-aims', action='store_true', default=False,
        help='write the final masks out as aims'
    )

    parser.add_argument(
        '--write-embeddings', action='store_true', default=False,
        help='write the final masks out as aims'
    )

    parser.add_argument(
        '--except_path', type=str, metavar="STR"
    )

    return parser


def main():
    print("Starting initialisation")
    parser = create_parser()
    args = parser.parse_args()

    # slice to plot
    plot_idx = 168 // 2

    # create predictions sub-directory
    pred_dir = os.path.join(args.data_dir, 'predictions_' + args.predictions_suffix)
    os.makedirs(pred_dir, exist_ok=True)

    if args.evaluate:
        eval_file = os.path.join(pred_dir, 'evaluation.csv')

    # check what device to use
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

    # create the model
    model = UNet(args.input_channels, args.output_channels,
                 args.model_filters, args.channels_per_group, args.dropout)
    model.float()
    model.to(device)

    # load the trained model parameters
    model.load_state_dict(torch.load(args.trained_model, map_location=device))

    # create dataset transforms
    data_transforms = Compose([
        SamplePadder(2 ** (len(args.model_filters) - 1)),
        SampleStandardizer(args.min_density, args.max_density),
        SampleToTensors(ohe=False)
    ])

    exceptions = PROBLEM_FILES
    if args.except_path:
        # load exceptions: files that should be skipped because they have been processed or are wrong
        exceptions_csv = pd.read_csv(args.except_path, delimiter=',')
        exceptions = exceptions_csv['name'].to_list() + exceptions

    # create dataset
    dataset = HRpQCTAIMDataset(args.data_dir, args.data_pattern, transform=data_transforms,
                               load_masks=args.load_ref_masks, exceptions=exceptions,
                               mask_type=args.predictions_suffix)

    # create kwargs for dataloader
    dataloader_kwargs = {
        'batch_size': 1,
        'shuffle': False
    }

    # createkwargs for dataset and dataloader for single images
    image_dataset_kwargs = {
        'num_adjacent_slices': (args.input_channels - 1) // 2
    }
    image_dataloader_kwargs = {
        'batch_size': 1,
        'shuffle': False
    }

    # create the overall dataloader
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # create a logger for dice scores if doingt that
    if args.evaluate:

        eval_metrics = ['dice', 'jaccard', 'ssd_max', 'ssd_mean']
        eval_masks = ['cort', 'trab']
        eval_methods = ['raw', 'post']

        eval_fields = ['name']
        for eval_metric in eval_metrics:
            for eval_mask in eval_masks:
                for eval_method in eval_methods:
                    eval_fields.append(f'{eval_metric}_{eval_mask}_{eval_method}')

        eval_logger = Logger(eval_file, eval_fields)

    # iterate through the images
    print("Initialised. Starting evaluation")
    for idx, image in enumerate(tqdm(dataloader)):

        labels = image['labels'][0, 0, :, :, :].cpu().detach().numpy()
        image_data = image['image'][0, 0, :, :, :].cpu().detach().numpy()

        cort_mask_reference = labels == 0
        if args.predictions_suffix == "xtremect1_periosteal":
            trab_mask_reference = cort_mask_reference
        else:
            trab_mask_reference = labels == 1

        image_name = os.path.splitext(image['name'][0])[0]

        phi_peri, phi_endo = infer(
            args, model, device, image,
            image_dataset_kwargs, image_dataloader_kwargs
        )

        cort_mask = (phi_peri < 0) * (phi_endo > 0)
        trab_mask = phi_endo < 0

        cort_mask_post, trab_mask_post = postprocess_masks_iterative(
            image_data, cort_mask, trab_mask, visualize=args.visualize, pred_dir=pred_dir, image_name=image_name
        )

        if args.predictions_suffix == "xtremect1_periosteal":
            cort_mask = trab_mask = cort_mask + trab_mask
            cort_mask_post = trab_mask_post = cort_mask_post + trab_mask_post

        if args.evaluate:
            print("Writing metrics")
            eval_logger.set_field_value('name', image_name)

            dice_cort_raw, jaccard_cort_raw = calculate_dice_and_jaccard(cort_mask, cort_mask_reference)
            dice_trab_raw, jaccard_trab_raw = calculate_dice_and_jaccard(trab_mask, trab_mask_reference)
            dice_cort_post, jaccard_cort_post = calculate_dice_and_jaccard(cort_mask_post, cort_mask_reference)
            dice_trab_post, jaccard_trab_post = calculate_dice_and_jaccard(trab_mask_post, trab_mask_reference)

            eval_logger.set_field_value('dice_cort_raw', dice_cort_raw)
            eval_logger.set_field_value('dice_trab_raw', dice_trab_raw)
            eval_logger.set_field_value('dice_cort_post', dice_cort_post)
            eval_logger.set_field_value('dice_trab_post', dice_trab_post)

            eval_logger.set_field_value('jaccard_cort_raw', jaccard_cort_raw)
            eval_logger.set_field_value('jaccard_trab_raw', jaccard_trab_raw)
            eval_logger.set_field_value('jaccard_cort_post', jaccard_cort_post)
            eval_logger.set_field_value('jaccard_trab_post', jaccard_trab_post)

            ssd_cort_raw = calculate_surface_distance_measures(cort_mask, cort_mask_reference,
                                                               [args.spacing, args.spacing, args.spacing])
            ssd_trab_raw = calculate_surface_distance_measures(trab_mask, trab_mask_reference,
                                                               [args.spacing, args.spacing, args.spacing])
            ssd_cort_post = calculate_surface_distance_measures(cort_mask_post, cort_mask_reference,
                                                                [args.spacing, args.spacing, args.spacing])
            ssd_trab_post = calculate_surface_distance_measures(trab_mask_post, trab_mask_reference,
                                                                [args.spacing, args.spacing, args.spacing])

            eval_logger.set_field_value('ssd_max_cort_raw', ssd_cort_raw['max'])
            eval_logger.set_field_value('ssd_max_trab_raw', ssd_trab_raw['max'])
            eval_logger.set_field_value('ssd_max_cort_post', ssd_cort_post['max'])
            eval_logger.set_field_value('ssd_max_trab_post', ssd_trab_post['max'])

            eval_logger.set_field_value('ssd_mean_cort_raw', ssd_cort_raw['mean'])
            eval_logger.set_field_value('ssd_mean_trab_raw', ssd_trab_raw['mean'])
            eval_logger.set_field_value('ssd_mean_cort_post', ssd_cort_post['mean'])
            eval_logger.set_field_value('ssd_mean_trab_post', ssd_trab_post['mean'])

            eval_logger.log()

        if args.write_embeddings:
            os.makedirs(f"{pred_dir}/embeddings/", exist_ok=True)
            np.savez_compressed(f"{pred_dir}/embeddings/{image_name}_embed.npz",
                                endosteal=phi_endo, periosteal=phi_peri)

        if args.write_aims:
            os.makedirs(f"{pred_dir}/masks/raw/", exist_ok=True)
            os.makedirs(f"{pred_dir}/masks/pp/", exist_ok=True)
            print("Writing AIMs")
            save_mask_as_AIM(
                f'{pred_dir}/masks/raw/{image_name}_CORT_MASK.AIM',
                cort_mask,
                image['image_position'],
                image['image_position_original'],
                image['image_shape_original'],
                image['spacing'],
                image['origin'],
                image['processing_log'][0],
                'Cortical mask',
                'UNet',
                'unreleased'
            )

            save_mask_as_AIM(
                f'{pred_dir}/masks/raw/{image_name}_TRAB_MASK.AIM',
                trab_mask,
                image['image_position'],
                image['image_position_original'],
                image['image_shape_original'],
                image['spacing'],
                image['origin'],
                image['processing_log'][0],
                'Trabecular mask',
                'UNet',
                'unreleased'
            )

            save_mask_as_AIM(
                f'{pred_dir}/masks/pp/{image_name}_CORT_MASK.AIM',
                cort_mask_post,
                image['image_position'],
                image['image_position_original'],
                image['image_shape_original'],
                image['spacing'],
                image['origin'],
                image['processing_log'][0],
                'Cortical mask',
                'UNet with post-processing',
                'unreleased'
            )

            save_mask_as_AIM(
                f'{pred_dir}/masks/pp/{image_name}_TRAB_MASK.AIM',
                trab_mask_post,
                image['image_position'],
                image['image_position_original'],
                image['image_shape_original'],
                image['spacing'],
                image['origin'],
                image['processing_log'][0],
                'Trabecular mask',
                'UNet with post-processing',
                'unreleased'
            )


if __name__ == '__main__':
    main()

import os

import numpy as np
import torch
from segmentation.tile_prediction import TilePrediction
from segmentation.unet import UNet
from skimage.morphology import remove_small_objects
from torch import nn

# args = {
#     "mask_magnification": 1.25,
#     "slide_dir": "E://_UNIVER/Diploma/data/data_leeds_sis",
#     "slide_id": "4796.sis",
#     "save_folder": "E://_UNIVER/Diploma/data/data_leeds_segmented",
#     "model": "E://_UNIVER/Diploma/PathProfiler/tissue_segmentation/checkpoint_147800.pth",
#     "tile_size": 512,
#     "batch_size": 1,
#     "mpp_level_0": 1
# }

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_CHECKPOINT = f"{DIR_PATH}/unet.pth"
TILE_SIZE = 512
BATCH_SIZE = 1
MPP_LEVEL_0 = 1
MASK_MAGNIFICATION = 1.25


def segment_slide(filename):
    unet = UNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = (
        nn.DataParallel(unet).cuda()
        if torch.cuda.is_available()
        else nn.DataParallel(unet)
    )
    print("=> loading checkpoint '{}'".format(MODEL_CHECKPOINT))
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    print(
        "=> loaded checkpoint '{}' (epoch {})".format(
            MODEL_CHECKPOINT, checkpoint["epoch"]
        )
    )

    net.eval()
    predictor = TilePrediction(
        patch_size=TILE_SIZE,
        subdivisions=2.0,
        pred_model=net,
        batch_size=BATCH_SIZE,
        workers=0,
        mask_magnification=MASK_MAGNIFICATION,
        mpp_level_0=MPP_LEVEL_0,
    )
    print("Processing", filename)
    try:
        segmentation = predictor.run(filename)
        segmentation = remove_small_objects(segmentation == 255, 50**2)
        segmentation = segmentation.astype(np.uint8)
        return segmentation
    except Exception as e:
        print(e, "\nSkipped slide", filename)
        return None

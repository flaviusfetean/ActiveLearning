import cv2
import numpy as np
import os
from imutils.paths import list_images

from segmentation.tests.al_sample_script import *
from unet import build_unet

def gen_overlays(IMAGES_PATH, MASKS_PATH, OVERLAYS_PATH):
    preds_names = list(list_images(MASKS_PATH))

    os.makedirs(OVERLAYS_PATH, exist_ok=True)

    for pred_path in preds_names:
        pred_name = pred_path.split(os.path.sep)[-1]
        image = cv2.imread(os.path.sep.join([IMAGES_PATH, pred_name]))
        mask = cv2.imread(os.path.sep.join([MASKS_PATH, pred_name]))

        image = cv2.resize(image, (mask.shape[1], mask.shape[0]))

        overlay = cv2.addWeighted(image, 0.75, mask, 0.25, 0)

        overlay = cv2.resize(overlay, (960, 512))

        cv2.imwrite(os.path.sep.join([OVERLAYS_PATH, pred_name]), overlay)


if __name__ == "__main__":
    IMAGES_PATH = "C:\\datasets\\Cityscapes\\imgs"
    MASKS_PATH = "C:\\Desktop\\predictions"
    OVERLAYS_PATH = "C:\\Desktop\\overlays"
    gen_overlays(IMAGES_PATH, MASKS_PATH, OVERLAYS_PATH)
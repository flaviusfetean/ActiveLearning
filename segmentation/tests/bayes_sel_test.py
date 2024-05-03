from typing import List

import tensorflow as tf
from unet import build_unet
from data_selectors import BayesianSelector, BaseSelector
from segmentation.selection_strategies.strategies import *
from al_sample_script import get_args, get_config, unet_preprocess_input
from data_handler import DataHandler

MODEL_WEIGHTS_PATH = r"C:\Github\active_learning\segmentation\tests\outputs\cityscapes_random\weights_full\model"
DATASET_PATH = r"C:\datasets\cityscapes\test\imgs"
LABELS_PATH = r"C:\datasets\cityscapes\test\masks_less"

def load_model(config):
    model = build_unet(input_size= (512, 960, 3), config=config)
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model


def main():
    args = get_args()
    configuration = get_config(args.config_file)
    h, w = configuration['img_h'], configuration['img_w']
    m, o = 76, 48
    model = load_model(configuration)
    data_handler = DataHandler(DATASET_PATH, LABELS_PATH)
    selector = BayesianSelector(AreaDisagreement(discrete=False), data_handler, 10, 10)
    #selector = BaseSelector(EntropySelection(), data_handler)
    selector.run(model, unet_preprocess_input(h, w, m, o))


if __name__ == "__main__":
    main()
from collections import namedtuple

import tqdm

from sklearn.model_selection import train_test_split
from utils.files_utils import get_paths, get_paths_recursive, create_folder, get_paths_recursive_with_extension
import shutil
import cv2
import numpy as np
import os

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label

    'lessId'      , # The id of the label in restricted classes mode
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color              mapped
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) , 0),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) , 0),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) , 0),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) , 0),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) , 0),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 1       , False        , True         , (111, 74,  0) , 0),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 2       , False        , True         , ( 81,  0, 81) , 0),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 3       , False        , False        , (128, 64,128)  , 1),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 4       , False        , False        , (244, 35,232)  , 2),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 5       , False        , True         , (250,170,160) , 2),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 6       , False        , True         , (230,150,140) , 0),
    Label(  'building'             , 11 ,        2 , 'construction'    , 7       , False        , False        , ( 70, 70, 70) , 0),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 8       , False        , False        , (102,102,156) , 0),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 9       , False        , False        , (190,153,153) , 0),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 10       , False        , True         , (180,165,180), 0),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 11       , False        , True         , (150,100,100), 0),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 12       , False        , True         , (150,120, 90), 0),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 13       , False        , False        , (153,153,153), 0),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 13       , False        , True         , (153,153,153), 0),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 14       , False        , False        , (250,170, 30), 3),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 15       , False        , False        , (220,220,  0), 4),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 16       , False        , False        , (107,142, 35), 0),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 17       , False        , False        , (152,251,152), 0),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 18       , False        , False        , ( 70,130,180), 0),
    Label(  'person'               , 24 ,       11 , 'human'           , 19       , True         , False        , (220, 20, 60), 5),
    Label(  'rider'                , 25 ,       12 , 'human'           , 20       , True         , False        , (255,  0,  0), 5),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 21       , True         , False        , (  0,  0,142), 6),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 22       , True         , False        , (  0,  0, 70), 6),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 23       , True         , False        , (  0, 60,100), 6),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 24       , True         , True         , (  0,  0, 90), 6),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 25       , True         , True         , (  0,  0,110), 6),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 26       , True         , False        , (  0, 80,100), 6),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 27       , True         , False        , (  0,  0,230), 7),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 28       , True         , False        , (119, 11, 32), 7),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0, 142), 6)
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# color to label_id
colorToLabel    = { label.color   : label.id for label in labels}
#label_id to color
LabelToColor    = { label.id      : [label.color[0], label.color[1], label.color[2]] for label in labels}

def Func_LabelToColorLess():
    l2c_less = {}
    for label in labels:
        l2c_less[label.lessId] = l2c_less.get(label.lessId, label.color)

    return l2c_less

LabelToColorLess = Func_LabelToColorLess()


#label id to color in restricted class mode
LessColorMap    = { label.id      : LabelToColorLess[label.lessId] for label in labels}


def unpack(images_path, masks_path, images_unpacked_path, masks_unpacked_path):
    images = get_paths_recursive(images_path)
    masks = get_paths_recursive_with_extension(masks_path, "_color.png")

    create_folder(images_unpacked_path)
    create_folder(masks_unpacked_path)

    for image_path, mask_path in tqdm.tqdm(zip(images, masks)):
        # image = cv2.imread(image_path)
        # mask = cv2.imread(mask_path)

        image_name = image_path.split(os.path.sep)[-1]
        mask_name = mask_path.split(os.path.sep)[-1]

        image_unpacked_path = os.path.sep.join([images_unpacked_path, image_name])
        mask_unpacked_path = os.path.sep.join([masks_unpacked_path, mask_name])

        shutil.copy(image_path, image_unpacked_path)
        shutil.copy(mask_path, mask_unpacked_path)


def extract_test(train_images_path, train_masks_path, test_images_path, test_masks_path, split_ratio=0.8):
    train_images = get_paths_recursive(train_images_path)
    train_masks = get_paths_recursive(train_masks_path)

    train_images, test_images, train_masks, test_masks = train_test_split(train_images, train_masks,
                                                                        test_size=1-split_ratio, random_state=42)

    create_folder(test_images_path)
    create_folder(test_masks_path)

    for image_path, mask_path in tqdm.tqdm(zip(test_images, test_masks)):
        image_name = image_path.split(os.path.sep)[-1]
        mask_name = mask_path.split(os.path.sep)[-1]

        test_image_path = os.path.sep.join([test_images_path, image_name])
        test_mask_path = os.path.sep.join([test_masks_path, mask_name])

        shutil.move(image_path, test_image_path)
        shutil.move(mask_path, test_mask_path)


def map_colors(masks_path, less_masks_path):
    all_masks = get_paths(masks_path)

    for mask_path in tqdm.tqdm(all_masks):
        name = mask_path.split(os.path.sep)[-1]
        if "color" in name:
            less_path = os.path.sep.join([less_masks_path, name])
            mask = cv2.imread(mask_path)[...,::-1]
            mask_less = np.zeros_like(mask)
            for id, color_less in LessColorMap.items():
                boolean_mask = np.all(mask == labels[id].color, axis=-1)
                mask_less[boolean_mask] = np.array(color_less, dtype=np.uint8)
            cv2.imwrite(less_path, mask_less)


if __name__ == "__main__":
    MASKS_PATH = "D:\\Deep_Learning_Projects\\datasets\\Cityscapes\\masks"
    LESS_MASKS_PATH = "D:\\Deep_Learning_Projects\\datasets\\Cityscapes\\train\\masks_less"
    IMAGES_PATH = "D:\\Deep_Learning_Projects\\datasets\\Cityscapes\\images"
    IMAGES_UNPACKED_PATH = "D:\\Deep_Learning_Projects\\datasets\\Cityscapes\\train\\images_unpacked"
    MASKS_UNPACKED_PATH = "D:\\Deep_Learning_Projects\\datasets\\Cityscapes\\train\\masks_unpacked"
    TEST_IMAGES_PATH = "D:\\Deep_Learning_Projects\\datasets\\Cityscapes\\test\\images_upacked"
    TEST_MASKS_LESS_PATH = "D:\\Deep_Learning_Projects\\datasets\\Cityscapes\\test\\masks_less"

    # with open("configs\\cityscapes_random_less.yaml", "a") as f:
    #     for color in LabelToColorLess.values():
    #         f.write(f"\n-{color}")

    # unpack(IMAGES_PATH, MASKS_PATH, IMAGES_UNPACKED_PATH, MASKS_UNPACKED_PATH)
    # map_colors(MASKS_UNPACKED_PATH, LESS_MASKS_PATH)
    extract_test(IMAGES_UNPACKED_PATH, LESS_MASKS_PATH, TEST_IMAGES_PATH, TEST_MASKS_LESS_PATH)

    # with open("configs\\cityscapes_random.yaml", "a") as f:
    #     for color in LabelToColor.values():
    #         f.write(f"\n-{color}")

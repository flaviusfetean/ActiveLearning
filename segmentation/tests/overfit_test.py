import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from unet import build_unet
from utils.files_utils import get_paths
from segmentation.utils import segmentation_preprocess, segmentation_preprocess_output
from al_sample_script import get_args, get_config


DATASET_PATH = "C:\\datasets\\Cityscapes\\imgs"
LABELS_PATH = "C:\\datasets\\Cityscapes\\masks"

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
        plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(unet_model, dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet_model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])



def overfit_test(model, configuration):
    image_paths = get_paths(configuration['images_path'])[:1]
    label_paths = get_paths(configuration['labels_path'])[:1]
    rgb_input = configuration['rgb_input']

    image = np.array([segmentation_preprocess(cv2.imread(image_path)[...,::-1], (480, 256), 76, 48) for image_path in image_paths]).astype("float32")
    label = np.array([segmentation_preprocess_output(cv2.imread(label_path)[..., :: 1 if rgb_input else -1], (480, 256), configuration['color_to_label'])for label_path in label_paths]).astype("float32")

    h, w, d = label[0].shape[0], label[0].shape[1], 3
    true_mask = np.zeros((h, w, d))
    pred_class = np.argmax(label[0], axis=-1)
    for class_index, color in configuration['label_to_color'].items():
        mask = pred_class == class_index
        true_mask[mask] = color


    model.fit(image, label, batch_size=1, epochs=500)

    prediction = model.predict(image)[0]

    h, w, d = prediction.shape[0], prediction.shape[1], 3
    pred_mask = np.zeros((h, w, d))
    pred_class = np.argmax(prediction, axis=-1)
    for class_index, color in configuration['label_to_color'].items():
        mask = pred_class == class_index
        pred_mask[mask] = color

    label = label[0]
    h, w, d = label.shape[0], label.shape[1], 3
    true_mask = np.zeros((h, w, d))
    pred_class = np.argmax(label, axis=-1)
    for class_index, color in configuration['label_to_color'].items():
        mask = pred_class == class_index
        true_mask[mask] = color

    cv2.imshow("predicted", pred_mask.astype('uint8'))
    cv2.imshow("true", true_mask.astype("uint8"))
    cv2.waitKey(0)

if __name__ == "__main__":

    args = get_args()
    configuration = get_config(args.config_file)
    model = build_unet(input_size=(256, 480, 3), config=configuration)

    overfit_test(model, configuration)
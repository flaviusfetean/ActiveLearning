import os
from typing import List, Callable

import cv2
import numpy as np
from utils.files_utils import get_paths_with_extensions
from bayesian_generator import aggregate_outputs, convert_to_probability
from scipy.stats import entropy
from segmentation.utils import miou, dice_loss

def get_bayesian_output(output_path: str):
    images_paths = get_paths_with_extensions(output_path, ['png', 'jpg'])
    predictions = np.array([cv2.imread(img_path) for img_path in images_paths])
    aggregate = np.array(aggregate_outputs(predictions, show=False), dtype="uint8")

    return predictions, aggregate

def score_entropy_simple(prediction: np.ndarray):
    entropies = []
    for channel in range(prediction.shape[-1]):
        pred_flattened = prediction.flatten()
        pred_normalized = pred_flattened / np.sum(pred_flattened)
        entropies.append(entropy(pred_normalized))

    return np.mean(entropies).item()

def score_entropy_of_hist(prediction: np.ndarray):
    entropies = []
    for channel in range(prediction.shape[-1]):
        hist = cv2.calcHist([prediction[:, :, channel]], [0], None, [256], [0, 256])
        hist /= hist.sum()
        entropies.append(entropy(hist))

    return np.mean(entropies).item()

def score_variance(prediction: np.ndarray):

    pred_flattened = prediction.flatten()
    stddev = pred_flattened.var()

    return np.asarray(stddev)

def score_avg_IoU(predictions: np.ndarray):

    predictions = predictions // 255.0
    predictions = np.expand_dims(predictions, axis=-1).astype("float32")

    ious = []
    for i in range(len(predictions) - 1):
        pd = predictions[i]
        for gt in predictions[i+1:]:
            ious.append(miou(gt, pd))

    return np.mean(ious).item()


def score_avg_dice_loss(predictions: np.ndarray):

    predictions = predictions / 255.0
    predictions = np.expand_dims(predictions, axis=-1).astype("float32")

    ious = []
    for i in range(len(predictions) - 1):
        pd = predictions[i]
        for gt in predictions[i+1:]:
            ious.append(dice_loss(gt, pd))

    return np.mean(ious).item()


def score_avg_dice_loss_wrt_mean(predictions: np.ndarray):
    predictions = predictions / 255.0 if np.any(predictions > 1) else predictions
    predictions = np.expand_dims(predictions, axis=-1).astype("float32")

    mean_prediction = aggregate_outputs(predictions).astype("float32")

    ious = []

    for prediction in predictions:
        ious.append(dice_loss(prediction.astype("float32"), mean_prediction))

    return np.mean(ious).item()

def display_collage(predictions: List[np.ndarray], names: List[str], score_func: Callable[[np.ndarray], float]):

    scores = [score_func(pred) for pred in predictions]

    name_height = 30
    image_height = predictions[0].shape[0]
    image_width = predictions[0].shape[1]
    text_height = 30
    collage_height = image_height + text_height + name_height
    collage_width = image_width * len(predictions)

    collage = np.ones((collage_height, collage_width, 3)) * 255

    w_offset = 0

    for name, prediction, score in zip(names, predictions, scores):
        try:
            collage[name_height:name_height + image_height, w_offset:w_offset+image_width, :] = prediction
        except ValueError:
            pass
        collage[:, w_offset, :] = 0
        cv2.putText(collage, f"score: {score}", (w_offset, collage_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(collage, name, (w_offset, name_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        w_offset += image_width


    cv2.imshow(score_func.__name__, collage.astype("uint8"))


if __name__ == "__main__":
    BASE_PATH = "segmentation\\tests\\simulation\\outputs_samples\\color_samples"
    dirs = os.listdir(BASE_PATH)

    outputs = [get_bayesian_output(os.path.sep.join([BASE_PATH,subdir])) for subdir in dirs]

    aggregates = [o[1] for o in outputs]
    all_preds = [o[0] for o in outputs]

    display_collage(aggregates, dirs, score_entropy_simple)
    display_collage(aggregates, dirs, score_entropy_of_hist)
    #display_collage(all_preds, dirs, score_avg_dice_loss_wrt_mean)
    display_collage(all_preds, dirs, score_avg_dice_loss_wrt_mean)
    #display_collage(all_preds, dirs, score_avg_IoU)
    #display_collage(all_preds, dirs, score_avg_dice_loss)

    cv2.waitKey(0)

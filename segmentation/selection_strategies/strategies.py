import random
from abc import ABC
from enum import Enum

import os
import cv2
from scipy.stats import entropy
from segmentation.utils import dice_coef, miou, sim_coef, dice_coef_supp
import numpy as np
import itertools

SMOOTH = os.environ.get("SMOOTH", False)

global_image_name = ""

def biggest_k_keys(dictionary: dict, k: int):
    sorted_dict = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    return dict(itertools.islice(sorted_dict.items(), k))


class SegmentationSelectionStrategy(ABC):
    """
    Selection strategy abstract class.
    """

    def score(self, prediction):
        """
        Metric of the selection strategy. The images will be selected based on this metric.
        The higher the value, higher the chance that the image should be selected.

        Args:
            prediction: Output of the specific model.

        Returns:
            Strategy score.

        """
        pass

    def get_best_samples(self, scores, selection_size=100):
        """
        Get the first ``selection_percentage`` * n samples based on the selection metric(strategy),
        where n is the dataset size.


        Args:
            scores: Dictionary with the outputs of the specific model. Keys: ids(image_names), values: predictions.
            selection_size: Positive value representing how many images should be selected in the next iteration.
        """
        assert 0 < selection_size <= len(scores), 'Selection size should be smaller than dataset size.'
        assert len(scores) > 0, 'Scores dictionary empty.'

        pass


class RandomSelection(SegmentationSelectionStrategy):
    def score(self, prediction):
        return None

    def get_best_samples(self, scores, selection_size=100):
        super().get_best_samples(scores, selection_size)
        sorted_scores = list(scores.keys())
        random.shuffle(sorted_scores)
        sorted_dict = {k: None for k in sorted_scores}
        return dict(itertools.islice(sorted_dict.items(), int(selection_size)))


class EntropySelection(SegmentationSelectionStrategy):
    def score(self, prediction):
        entropy_map = entropy(prediction, axis=3)
        return float(np.sum(entropy_map))

    def get_best_samples(self, scores, selection_size=100):
        super().get_best_samples(scores, selection_size)
        return biggest_k_keys(scores, selection_size)


class MarginSelection(SegmentationSelectionStrategy):
    def score(self, prediction):
        # orders predictions depthwise for each pixel
        ordered_preds = np.sort(prediction, axis=-1)
        ordered_preds_f2 = ordered_preds[:, :, :, -2:] # takes highest 2 predictions
        # multiply first entry with -1 (first entry is second-highest pred)
        ordered_preds_f2[:, :, :, 0] *= -1.0
        # sum the preds depthwise (will result in difference between highest 2 preds)
        margins = np.sum(ordered_preds_f2, axis=-1)
        # return with - as we want to minimize difference between highest 2 predictions, hence maximize negative difference
        return -float(np.sum(margins, axis=(1, 2)))

    def get_best_samples(self, scores, selection_size=100):
        super().get_best_samples(scores, selection_size)
        return biggest_k_keys(scores, selection_size)


label_to_word = {0: 'background', 1: 'road', 2: 'sidewalk',
                 3: 'traffic light', 4: 'traffic sign',
                 5: 'person', 6: 'vehicle', 7: 'two wheeler'}


class UncertaintySelection(SegmentationSelectionStrategy):
    def score(self, prediction):
        """
        prediction: In the case of disagreement, we suppose that we have an array of
                    multiple predictions coming from different parameter sets sampled
                    from a parameter distribution (i. e: different Monte-Carlo iterations)
                    prediction = num_samples x height x width x num_classes
        """
        assert len(prediction) > 1, 'Cannot compute uncertainty metric on a single prediction'
        entropies = np.nan_to_num(entropy(prediction, axis=0), nan=0)[0]

        entropies_norm = np.abs((128 - (entropies / np.max(entropies)) * 255) * 2).astype(np.uint8)
        for channel in range(entropies_norm.shape[-1]):
            chn_uncert = entropies_norm[..., channel]
            cv2.imshow(f"{label_to_word[channel]}", chn_uncert)

        uncertainty = np.mean(entropies, axis=-1)

        # uncertainty = np.abs((128 - uncertainty / np.max(uncertainty) * 255) * 2).astype(np.uint8)
        #
        # per_class_entropies = np.sum(entropies, axis=(0, 1))
        # print(per_class_entropies)
        # weighted_uncertainty = np.mean(per_class_entropies)
        # cv2.imshow(f"{weighted_uncertainty}", uncertainty)
        # cv2.waitKey(0)

        return float(np.sum(uncertainty, axis=(0, 1)))

    def get_best_samples(self, scores, selection_size=100):
        super().get_best_samples(scores, selection_size)
        return biggest_k_keys(scores, selection_size)


class DisagreementSelection(SegmentationSelectionStrategy):
    def score(self, prediction):
        """
        prediction: In the case of disagreement, we suppose that we have an array of
                    multiple predictions coming from different parameter sets sampled
                    from a parameter distribution (i. e: different Monte-Carlo iterations)
                    prediction = num_samples x height x width x num_classes
        """
        assert len(prediction) > 1, 'Cannot compute disagreement metric on a single prediction'
        ensemble_prediction_entropy = entropy(np.mean(prediction, axis=0), axis=-1)[0]
        mean_individual_entropy = np.mean(np.nan_to_num(entropy(prediction, axis=-1), nan=0), axis=0)[0]
        pixelwise_mutual_info = ensemble_prediction_entropy - mean_individual_entropy

        pixelwise_mutual_info = np.abs((128 - pixelwise_mutual_info / np.sum(pixelwise_mutual_info) * 255) * 2).astype(np.uint8)
        cv2.imshow(f"{float(np.sum(pixelwise_mutual_info, axis=(0, 1)))}", pixelwise_mutual_info)
        cv2.waitKey(0)

        return float(np.sum(pixelwise_mutual_info, axis=(0, 1)))

    def get_best_samples(self, scores, selection_size=100):
        super().get_best_samples(scores, selection_size)
        return biggest_k_keys(scores, selection_size)


def temp_print_vals(aggr: np.ndarray, image_name):
    imgs = []
    lmaps = []
    for label in range(aggr.shape[-1]):
        label_map = 255 - np.abs(128 - aggr[...,label] * 255) * 2
        #label_map = aggr[..., label] * 255
        label_map = np.clip(label_map, 0, 255)
        lmaps.append(label_map)
        img = np.dstack([label_map, label_map, label_map]).astype(np.uint8)
        cv2.imwrite(
            f"C:\\Desktop\\lic\\real_disagreement\\1_epoch_trained\\uncert_map_{label_to_word[label]}_{image_name[:-4]}.png",
            img)

        imgs.append(cv2.resize(img, (480, 256), cv2.INTER_NEAREST))
        #cv2.imshow(label_to_word[label], img)

    aggr_mean = np.mean(np.array(lmaps), axis=0)
    #cv2.imshow("mean", aggr_mean)

    return np.hstack(imgs)

class AreaDisagreement(SegmentationSelectionStrategy):
    def __init__(self, discrete):
        self.discrete = discrete

    def score(self, prediction, image_name=""):

        if self.discrete:
            num_labels = prediction.shape[-1]
            labels = np.argmax(prediction, axis=-1)
            prediction = np.eye(num_labels)[labels]

        aggregate = np.average(np.array(prediction), axis=0)[0]
        disagreements = []
        for pred_instance in prediction:
            if SMOOTH:
                disagreements.append(dice_coef(aggregate, pred_instance, smooth=1000))
            else:
                disagreements.append(dice_coef_supp(aggregate, pred_instance, smooth=0.1))
            # for label in range(prediction.shape[-1]):
            #     label_pred = pred_instance[...,label]
            #     print(f"dice coef {label}: {dice_coef(np.expand_dims(aggregate[..., label],axis=-1), np.expand_dims(label_pred,axis=-1), smooth=0.1)}")

        aggrs = temp_print_vals(aggregate, image_name)
        #cv2.imshow(str(np.mean(disagreements).item()), aggrs)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return -np.mean(disagreements).item()

    def get_best_samples(self, scores, selection_size=100):
        super().get_best_samples(scores, selection_size)
        return biggest_k_keys(scores, selection_size)


class AreaDisagreementEntropy(SegmentationSelectionStrategy):
    def __init__(self, discrete):
        self.discrete = discrete

    def score(self, prediction):

        if self.discrete:
            num_labels = prediction.shape[-1]
            labels = np.argmax(prediction, axis=-1)
            prediction = np.eye(num_labels)[labels]

        aggregate = np.average(np.array(prediction), axis=0)
        entropies = []
        for channel in range(aggregate.shape[-1]):
            pred_flattened = aggregate.flatten()
            pred_normalized = pred_flattened / np.sum(pred_flattened)
            entropies.append(entropy(pred_normalized))

        aggrs = temp_print_vals(aggregate)
        cv2.imshow(str(np.mean(entropies).item()), aggrs)

        return np.mean(entropies).item()

    def get_best_samples(self, scores, selection_size=100):
        super().get_best_samples(scores, selection_size)
        return biggest_k_keys(scores, selection_size)


class AreaDisagreementHistEntropy(SegmentationSelectionStrategy):
    def score(self, prediction):
        num_labels = prediction.shape[-1]
        labels = np.argmax(prediction, axis=-1)
        prediction = np.eye(num_labels)[labels]

        entropies = []
        for channel in range(prediction.shape[-1]):
            pred_flattened = prediction.flatten()
            pred_normalized = pred_flattened / np.sum(pred_flattened)
            entropies.append(entropy(pred_normalized))


        return np.mean(entropies).item()

    def get_best_samples(self, scores, selection_size=100):
        super().get_best_samples(scores, selection_size)
        return biggest_k_keys(scores, selection_size)


class StrategyClass(Enum):
    entropy = EntropySelection()
    random = RandomSelection()
    margin = MarginSelection()
    uncertainty = UncertaintySelection()
    disagreement = DisagreementSelection()
    area = AreaDisagreement(discrete=False)

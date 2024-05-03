import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K

def my_cce(y_true, y_pred):
    clip = 0.0001
    normalizer = tf.math.reduce_prod(tf.cast(tf.shape(y_pred)[:-1], dtype=tf.float32))
    y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
    return -tf.reduce_sum(y_true * tf.where(y_pred != .0, tf.math.log(y_pred + clip), .0)) / normalizer


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gamma=1):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(y_pred ** self.gamma) + tf.reduce_sum(y_true ** self.gamma) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result

def categorical_focal_loss(alpha=0.25, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        epsilon = 0.0001
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

def dice_coef_supp(y_true, y_pred, smooth=0.00001):
    intersection = 2 * tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0, 1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2])
    normalizer = tf.reduce_sum(tf.clip_by_value(intersection + union, 0, 1))
    dice = (intersection + smooth) / (union + smooth)
    dice = tf.where(dice > 0.4, dice, 1.0)
    dice = tf.reduce_sum(dice) / normalizer
    return dice

def dice_coef(y_true, y_pred, smooth=0.00001):
    intersection = 2 * tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0, 1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2])
    normalizer = tf.reduce_sum(tf.clip_by_value(intersection + union, 0, 1))
    dice = (intersection + smooth) / (union + smooth)
    #dice = tf.where(dice > 0.2, dice, 1.0)
    dice = tf.reduce_sum(dice) / normalizer
    return dice

def sim_coef(y_true, y_pred, smooth=0.00001):
    difference = tf.reduce_sum(tf.abs(y_pred - y_true), axis=[0, 1, 2])
    intersection = 2 * tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0, 1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2])
    normalizer = tf.reduce_sum(tf.clip_by_value(intersection + union, 0, 1))
    sim = tf.maximum(difference + smooth, 0) / (union + smooth)
    sim = tf.reduce_sum(sim) / normalizer
    return sim

def dice_loss(y_true, y_pred, eps=0.00001):
    return 1. - dice_coef(y_true, y_pred, eps)


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def dice_loss_np(y_true, y_pred, eps=10e-5):
    y_true_f = np.flatten(y_true)
    y_pred_f = np.flatten(y_pred)

    inter_area = np.sum(y_pred_f * y_true_f)
    sum_areas = np.sum(y_true_f + y_pred_f)
    return 1 - 2 * (inter_area + eps) / (sum_areas + eps)


def miou(y_true, y_pred, eps=10e-5):
    preds_argmax = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(preds_argmax, y_pred.shape[-1], axis=-1)

    inter_area = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0, 1, 2])
    #tf.print(inter_area)
    sum_areas = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2])
    #tf.print(sum_areas)
    union = sum_areas - inter_area

    normalizer = tf.reduce_sum(tf.clip_by_value(sum_areas, 0, 1))
    #tf.print(normalizer)

    iou_comp = (inter_area + eps) / (union + eps)
    iou_comp = tf.reduce_sum(iou_comp) / normalizer

    return iou_comp

def iou(y_true, y_pred, eps=10e-5):
    preds_argmax = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(preds_argmax, y_pred.shape[-1], axis=-1)

    inter_area = tf.reduce_sum(tf.multiply(y_true, y_pred))
    sum_areas = tf.reduce_sum(y_true + y_pred)
    union = sum_areas - inter_area

    normalizer = tf.reduce_sum(tf.clip_by_value(sum_areas, 0, 1))

    iou_comp = (inter_area + eps) / (union + eps)
    iou_comp = iou_comp / normalizer

    return iou_comp


def segmentation_preprocess(image, image_size, mean, stddev) -> np.ndarray:
    image = cv2.resize(image, image_size)
    image = (image.astype('float32') - mean) / stddev
    return image


def tf_segmentation_preprocess(image, image_size, mean, stddev):
    image = tf.image.resize(image, image_size)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.divide(tf.subtract(image, mean), stddev)
    return image


def segmentation_preprocess_output(mask, image_size, color_to_label: dict) -> np.ndarray:
    mask_array = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
    categ = mask_to_categorical(mask_array, color_to_label)
    return categ


def tf_segmentation_preprocess_output(mask, image_size, color_to_label: dict):
    mask_array = tf.image.resize(mask, image_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    categ = tf_mask_to_categorical(mask_array, color_to_label)
    return categ


def categorical_to_mask(prediction, label_to_color: dict):
    h, w, d = prediction.shape[0], prediction.shape[1], 3
    mask_color = np.zeros((h, w, d))
    pred_class = np.argmax(prediction, axis=-1)
    for class_index, color in label_to_color.items():
        mask = pred_class == class_index
        mask_color[mask] = color

    return mask_color


def mask_to_categorical(mask_array, color_to_label: dict):
    class_indices = np.zeros(mask_array.shape[:2], dtype=np.uint8)
    mask_array = np.array(mask_array, dtype=np.uint8)
    for color, class_index in color_to_label.items():
        color_vector = np.array(color, dtype=np.uint8)
        mask = np.all(mask_array == color_vector, axis=-1)
        class_indices[mask] = class_index

    # Create a one-hot encoded representation using np.eye()
    num_classes = len(color_to_label)
    categ = np.eye(num_classes)[class_indices]

    return categ

def tf_mask_to_categorical(mask_array, color_to_label: dict):
    mask_array = tf.convert_to_tensor(mask_array)
    one_hot_map = []
    for colour in color_to_label:
        class_map = tf.reduce_all(tf.equal(mask_array, colour), axis=-1)
        one_hot_map.append(class_map)

    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.float32)


    #tf.print(one_hot_map)

    return one_hot_map
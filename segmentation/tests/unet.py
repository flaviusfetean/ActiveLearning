"""
    @article{Maynard-Reid_2022_U-Net,
      author = {Margaret Maynard-Reid},
      title = {{U-Net} Image Segmentation in Keras},
      journal = {PyImageSearch},
      year = {2022},
      note = {https://pyimg.co/6m5br},
    }
    U-Net implementation copied from https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
    for fast testing of the methods in the beginning
"""
import keras.metrics
import tensorflow as tf
from keras import layers
from segmentation.utils import iou, miou, dice_coef, dice_loss, categorical_focal_loss, my_cce

NAME_TO_LOSS = {
    'crossentropy': my_cce,
    'dice_loss': dice_loss,
    'focal_loss': categorical_focal_loss(),
}

def double_conv_block(x, n_filters, batch_norm=False):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, kernel_size=(3, 3), padding="same", activation="relu")(
        x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, kernel_size=(3, 3), padding="same", activation="relu")(
        x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    return x


def downsample_block(x, n_filters, dropout):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D((2, 2))(f)
    if dropout > 0:
        p = layers.Dropout(dropout)(p)
    return f, p


def upsample_block(x, conv_features, n_filters, dropout, batch_norm):
    # upsample
    x = layers.Conv2DTranspose(n_filters, (3, 3), (2, 2), padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters, batch_norm)
    return x


def build_unet(input_size=(960, 512, 3), config = None):

    assert config is not None, 'No Model configuration proided! Take the example configuration showwn in segmentation/tests/configs/*.yaml'

    num_classes = config['num_classes']
    num_filters = config['num_filters']


    inputs = layers.Input(shape=input_size)
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, num_filters, config['dropout'])
    # 2 - downsample
    f2, p2 = downsample_block(p1, 2 * num_filters, config['dropout'])
    # 3 - downsample
    f3, p3 = downsample_block(p2, 4 * num_filters, config['dropout'])
    # 4 - downsample
    f4, p4 = downsample_block(p3, 8 * num_filters, config['dropout'])
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 16 * num_filters, config['batch_norm'])
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 8 * num_filters, config['dropout'], config['batch_norm'])
    # 7 - upsample
    u7 = upsample_block(u6, f3, 4 * num_filters, config['dropout'], config['batch_norm'])
    # 8 - upsample
    u8 = upsample_block(u7, f2, 2 * num_filters, config['dropout'], config['batch_norm'])
    # 9 - upsample
    u9 = upsample_block(u8, f1, num_filters, config['dropout'], config['batch_norm'])
    # outputs
    outputs = layers.Conv2D(num_classes, (1, 1), padding="same", activation="softmax")(u9)
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    if config['loss'] not in NAME_TO_LOSS:
        print("[WARNING] Invalid loss function specified. Defaulting to cross-entropy")

    unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr_init']),
                       loss=NAME_TO_LOSS[config['loss']],
                       metrics=[miou, iou, dice_coef])

    #unet_model.summary()

    return unet_model

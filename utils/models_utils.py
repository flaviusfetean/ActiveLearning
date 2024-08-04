import tensorflow as tf


def create_model_with_dropout_heads(model, dropout_rate=0.5, num_heads=10):
    """
    Adds dropout heads to a model before the last parameterized layer, by branching from there
    Args:
        model: tf.keras.Model
            the model body to branch its body with dropout heads
        dropout_rate: float
            the dropout rate to apply to the dropout heads
        num_heads: int
            the number of dropout heads to add to the model

    Returns: tf.keras.Model
        the model with the dropout heads branching frm its body

    """

    # Find the last parameterized layer
    last_param_layer_index = None
    for i, layer in enumerate(model.layers):
        # Check if the layer has trainable weights and a kernel attribute (dot product with weights)
        if layer.trainable and hasattr(layer, 'kernel'):
            last_param_layer_index = i

    if last_param_layer_index is None:
        raise ValueError("No parameterized layer found in the model.")

    # Create a new body by copying layers up to the last parameterized layer
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs
    for i in range(last_param_layer_index):
        x = model.layers[i](x)

    # Create the dropout heads
    dropout_heads = []

    for i in range(num_heads):
        output_dropout = tf.keras.layers.Dropout(0.5)(x, training=True)
        # Add the remaining layers after the dropout
        for j in range(last_param_layer_index, len(model.layers)):
            output_dropout = model.layers[j](output_dropout)
        dropout_heads.append(output_dropout)

    # Create a new model
    new_model = tf.keras.Model(inputs=inputs, outputs=dropout_heads)
    return new_model


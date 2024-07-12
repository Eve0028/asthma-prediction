import tensorflow as tf

from params import EMBEDDING_SIZE, dropout_rate1, fc_neurons, reg_l2, dropout_rate2, sample_width, sample_height


# Modify VGGish to use TimeDistributed layers and add custom layers
def create_time_distributed_vggish():
    inputs = tf.keras.Input(shape=(None, sample_width, sample_height))
    # Add the channel (for conv2d layers)
    x = tf.keras.layers.Lambda(lambda y: tf.expand_dims(y, axis=-1))(inputs)

    # 1st Conv block
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1/conv1_1')
    )(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')
    )(x)

    # 2nd Conv block
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2/conv2_1')
    )(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')
    )(x)

    # 3rd Conv block
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3/conv3_1')
    )(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3/conv3_2')
    )(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')
    )(x)

    # 4th Conv block and flatten
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4/conv4_1')
    )(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4/conv4_2')
    )(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')
    )(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten(name='flatten')
    )(x)

    # Fully connected layers
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(4096, activation='relu', name='fc1/fc1_1')
    )(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(4096, activation='relu', name='fc1/fc1_2')
    )(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(EMBEDDING_SIZE, activation='relu', name='fc2')
    )(x)

    # Dropout
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dropout(dropout_rate1)
    )(x)
    # 1st additional fully connected layer
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(fc_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_l2),
                              name='fc3')
    )(x)

    # Dropout
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dropout(dropout_rate2)
    )(x)
    # 2nd additional fully connected layer (logits)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(2, activation=None, kernel_regularizer=tf.keras.regularizers.l2(reg_l2), name='logits')
    )(x)

    # Global Average Pooling
    x = tf.keras.layers.Activation('softmax')(x)
    outputs = tf.keras.layers.GlobalAveragePooling1D()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def load_vggish_weights(model, vggish_checkpoint_path):
    vggish_layers = {
        'conv1/conv1_1': ('vggish/conv1/weights', 'vggish/conv1/biases'),
        'conv2/conv2_1': ('vggish/conv2/weights', 'vggish/conv2/biases'),
        'conv3/conv3_1': ('vggish/conv3/conv3_1/weights', 'vggish/conv3/conv3_1/biases'),
        'conv3/conv3_2': ('vggish/conv3/conv3_2/weights', 'vggish/conv3/conv3_2/biases'),
        'conv4/conv4_1': ('vggish/conv4/conv4_1/weights', 'vggish/conv4/conv4_1/biases'),
        'conv4/conv4_2': ('vggish/conv4/conv4_2/weights', 'vggish/conv4/conv4_2/biases'),
        'fc1/fc1_1': ('vggish/fc1/fc1_1/weights', 'vggish/fc1/fc1_1/biases'),
        'fc1/fc1_2': ('vggish/fc1/fc1_2/weights', 'vggish/fc1/fc1_2/biases'),
        'fc2': ('vggish/fc2/weights', 'vggish/fc2/biases')
    }

    reader = tf.train.load_checkpoint(vggish_checkpoint_path)

    for layer_name, (weights_key, biases_key) in vggish_layers.items():
        weights = reader.get_tensor(weights_key)
        biases = reader.get_tensor(biases_key)
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.TimeDistributed) and layer.layer.name == layer_name:
                layer.layer.set_weights([weights, biases])
                # wghts = layer.layer.get_weights()
                break


def get_model():
    time_distributed_vggish = create_time_distributed_vggish()

    # Load pre-trained VGGish weights
    vggish_checkpoint_path = 'vggish_model.ckpt'
    load_vggish_weights(time_distributed_vggish, vggish_checkpoint_path)

    return time_distributed_vggish

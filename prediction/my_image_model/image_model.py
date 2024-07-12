import tensorflow as tf


def architecture(base_model_trained, input_shape=(224, 224, 3), num_units=128, num_after_pooling_units=2048,
                 reg_l2=1e-6, dropout=0.5, base_model_trainable=False):
    # create the base pre-trained model
    base_model = base_model_trained(weights='imagenet',
                                    include_top=False,
                                    input_shape=input_shape)

    # If necessary - freeze the base_model
    base_model.trainable = base_model_trainable

    # Create new model on top
    inputs = tf.keras.layers.Input(shape=input_shape)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.reduce_mean(input_tensor=x, axis=0)
    x = tf.reshape(x, (-1, num_after_pooling_units), name="mean_reshape")

    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(num_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs, predictions)

    return base_model, model

# coding: utf-8
# author: T.XIA
# editor: Eve

"""Defines the 'audio' model used to classify the VGGish features."""

from __future__ import print_function

import sys

import tensorflow as tf
import tf_slim as slim

import model_params as params
import prediction.vggish.vggish_slim  # noqa: E402

sys.path.append("../vggish")


def define_audio_slim(
        reg_l2=params.L2,
        num_units=params.NUM_UNITS,
        train_vgg=False
):
    """Defines the audio TensorFlow model.

    All ops are created in the current default graph, under the scope 'audio/'.

    The input is a placeholder named 'audio/vggish_input' of type float32 and
    shape [batch_size, feature_size] where batch_size is variable and
    feature_size is constant, and feature_size represents a VGGish output feature.
    The output is an op named 'audio/prediction' which produces the activations of
    a NUM_CLASSES layer.

    Args:
        training: If true, all parameters are marked trainable.
        :param train_vgg: If true, VGGish parameters are trainable.
        :param reg_l2:
        :param num_units:

    Returns:
        The op 'mymodel/logits'.
    """

    embeddings = prediction.vggish.vggish_slim.define_vggish_slim(
        train_vgg)  # (? x 128) vggish is the pre-trained model
    print("model summary:", train_vgg)

    with slim.arg_scope(
            [slim.fully_connected],
            weights_initializer=tf.compat.v1.truncated_normal_initializer(
                stddev=params.INIT_STDDEV, seed=0
            ),  # 1 is the best for old data
            biases_initializer=tf.compat.v1.zeros_initializer(),
            weights_regularizer=tf.keras.regularizers.l2(0.5 * (reg_l2)),
    ), tf.compat.v1.variable_scope("mymodel"):
        # index = tf.compat.v1.placeholder(dtype=tf.int32, shape=(1, 1), name="index")  # split B C V
        # index2 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(1, 1), name="index2")
        dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_rate")

        with tf.compat.v1.name_scope("Breath"):
            # fc_vgg_breath = embeddings[0: index[0, 0], :]  # (len, 128)
            fc1_b = tf.reduce_mean(input_tensor=embeddings, axis=0)
            fc2_b = tf.reshape(fc1_b, (-1, 128), name="vgg_b")

        with tf.compat.v1.name_scope("Output"):
            # classifier
            # fc3 = fc2_b

            # classification
            fc3_dp = tf.nn.dropout(fc2_b, rate=1 - (dropout_keep_prob[0, 0]), seed=0)
            fc4 = slim.fully_connected(fc3_dp, num_units)
            fc4_dp = tf.nn.dropout(fc4, rate=1 - (dropout_keep_prob[0, 0]), seed=0)
            logits = slim.fully_connected(fc4_dp, params.NUM_CLASSES, activation_fn=None, scope="logits")
            tf.nn.softmax(logits, name="prediction")
        return logits


def load_audio_slim_checkpoint(session, checkpoint_path):
    """Loads a pre-trained audio-compatible checkpoint.

    This function can be used as an initialization function (referred to as
    init_fn in TensorFlow documentation) which is called in a Session after
    initializating all variables. When used as an init_fn, this will load
    a pre-trained checkpoint that is compatible with the audio model
    definition. Only variables defined by audio will be loaded.

    Args:
        session: an active TensorFlow session.
        checkpoint_path: path to a file containing a checkpoint that is
          compatible with the audio model definition.
    """

    # Get the list of names of all audio variables that exist in
    # the checkpoint (i.e., all inference-mode audio variables).
    with tf.Graph().as_default():
        define_audio_slim()
        audio_var_names = [v.name for v in tf.compat.v1.global_variables()]

    # Get list of variables from exist graph which passed by session
    with session.graph.as_default():
        global_variables = tf.compat.v1.global_variables()

    # Get the list of all currently existing variables that match
    # the list of variable names we just computed.
    audio_vars = [v for v in global_variables if v.name in audio_var_names]

    # Use a Saver to restore just the variables selected above.
    saver = tf.compat.v1.train.Saver(audio_vars, name="audio_load_pretrained", write_version=1)
    saver.restore(session, checkpoint_path)

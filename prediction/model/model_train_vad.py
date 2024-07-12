# -*- coding: utf-8 -*-
"""
Created on Wed OCt 9 17:18:28 2020

@author: XT
@editor: Eve
"""
from __future__ import print_function

import argparse
import datetime
import os
import random
import sys

import numpy as np
import tensorflow as tf
import tf_slim as slim
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
tf.compat.v1.set_random_seed(SEED)

NNI = False

import model_util as util  # noqa: E402
from model_network import define_audio_slim, params  # noqa: E402

sys.path.append("../vggish")
import warnings  # noqa: E402

from prediction.vggish.vggish_slim import load_vggish_slim_checkpoint  # noqa: E402

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--train_name", type=str, default=params.AUDIO_TRAIN_NAME, help="Name of this programe.")
parser.add_argument("--task", type=str, default=params.TASK, help="Name of the task.")
parser.add_argument("--data_name", type=str, default=params.DATA_NAME, help="Original data path.")
parser.add_argument("--is_aug", type=bool, default=False, help="Add data augmentation.")
parser.add_argument("--restore_if_possible", type=bool, default=False, help="Restore variables.")
parser.add_argument("--epoch_to_restore", type=int, default=False, help="Epoch of model weights in filename.")
parser.add_argument("--epoch", type=int, default=100, help="Maximum epoch to train.")
parser.add_argument("--epoch_to_test", type=int, default=False, help="Epoch in with model test on test data.")
parser.add_argument(
    "--early_stop", type=str, default=params.EARLY_STOP, help="The indicator on validation set to stop training."
)
parser.add_argument("--is_diff", type=bool, default=True, help="Whether to use differential learing rate.")
parser.add_argument("--train_vgg", type=bool, default=True, help="Fine tuning Vgg")
parser.add_argument("--enable_checkpoint", type=bool, default=True, help="Use pretrained VGGish.")

parser.add_argument("--reg_l2", type=float, default=params.L2, help="L2 regulation.")
parser.add_argument("--lr_decay", type=float, default=params.LEARNING_RATE_DECAY, help="learning rate decay rate.")
parser.add_argument("--dropout_rate", type=float, default=params.DROPOUT_RATE, help="Dropout rate.")

parser.add_argument(
    "--trained_layers", type=int, default=params.TRAINED_LAYERS, help="The number Vgg layers to be fine tuned."
)
parser.add_argument("--num_units", type=int, default=64, help="The numer of unit in network.")
parser.add_argument("--lr1", type=float, default=1e-4, help="learning rate for Vgg layers.")
parser.add_argument("--lr2", type=float, default=1e-4, help="learning rate for top layers.")

# parser.add_argument("--adam_eps", type=float, default=1e-8, help="Epsilon for the Adam optimizer.")
# parser.add_argument("--init_stddev", type=float, default=0.01, help="Standard deviation used to initialize weights.")

FLAGS, _ = parser.parse_known_args()
FLAGS = vars(FLAGS)

data_dir = os.path.join(params.TF_DATA_DIR, FLAGS["data_name"])  # ./data
tensorboard_dir = os.path.join(params.TENSORBOARD_DIR, FLAGS["train_name"])  # ./data/tensorbord/
audio_ckpt_dir = os.path.join(
    params.AUDIO_CHECKPOINT_DIR, FLAGS["train_name"]
)  # ./data/train/ name_modality: name, with/out feature, modality: B
name_pre = (
        "Drp"
        + str(FLAGS["dropout_rate"])
        + "_"
        + "Uni"
        + str(FLAGS["num_units"])
        + "_"
        + "L2"
        + str(FLAGS["reg_l2"])
        + "_"
        + "VGGL"
        + str(FLAGS["trained_layers"])
)
name_mid = "DC" + str(FLAGS["lr_decay"]) + "_" + "LR" + str(FLAGS["lr1"]) + "_" + str(FLAGS["lr2"])
name_all = name_pre + "__" + name_mid + "__"
print("save:", name_all)

util.maybe_create_directory(tensorboard_dir)
util.maybe_create_directory(audio_ckpt_dir)


def model_summary():
    """Print model to log."""
    print("\n")
    print("=" * 30 + "Model Structure" + "=" * 30)
    model_vars = tf.compat.v1.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    print("=" * 60 + "\n")


def _create_data():
    """Create audio `train`, `test` and `val` records file."""
    tf.compat.v1.logging.info("Create records..")
    _check_vggish_ckpt_exists()
    train, val, test = util.load_data(data_dir, samples_count=None)
    tf.compat.v1.logging.info("Dataset size: Train-{} Test-{} Val-{}".format(len(train), len(test), len(val)))
    return train, val, test


def _add_triaining_graph():
    """Define the TensorFlow Graph."""
    with tf.Graph().as_default() as graph:
        logits = define_audio_slim(
            reg_l2=FLAGS["reg_l2"],
            num_units=FLAGS["num_units"],
            train_vgg=FLAGS["train_vgg"],
        )
        # tf.compat.v1.summary.histogram("logits", logits)
        # define training subgraph
        with tf.compat.v1.variable_scope("train"):
            labels = tf.compat.v1.placeholder(tf.float32, shape=[None, params.NUM_CLASSES], name="labels")
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(labels),
                                                                    name="cross_entropy")
            cla_loss = tf.reduce_mean(input_tensor=cross_entropy, name="cla_loss")
            reg_loss2 = tf.add_n(
                [tf.nn.l2_loss(v) * FLAGS["reg_l2"] for v in tf.compat.v1.trainable_variables() if
                 "bias" not in v.name],
                name="reg_loss2",
            )
            loss = tf.add(reg_loss2, cla_loss, name="loss_op")

            # tf.compat.v1.summary.scalar("loss", loss)
            # training
            global_step = tf.compat.v1.Variable(
                0,
                name="global_step",
                trainable=False,
                collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, tf.compat.v1.GraphKeys.GLOBAL_STEP],
            )

            # Use Learning Rate Decaying for top layers
            number_decay_steps = 7000 if FLAGS["is_aug"] else 30000  # approciately an epoch
            base_of = FLAGS["lr_decay"]
            lr1 = tf.compat.v1.train.exponential_decay(
                FLAGS["lr1"], global_step, number_decay_steps, base_of, staircase=True, name="train_lr1"
            )
            lr2 = tf.compat.v1.train.exponential_decay(
                FLAGS["lr2"], global_step, number_decay_steps, base_of, staircase=True, name="train_lr2"
            )

            if FLAGS["is_diff"]:  # use different learning rate for vgg and others
                print("--------------learning rate control-----------------")
                var1 = tf.compat.v1.trainable_variables()[
                       params.VGGISH_CNT_TRAINABLE - FLAGS["trained_layers"]:params.VGGISH_CNT_TRAINABLE]  # Vggish
                var2 = tf.compat.v1.trainable_variables()[params.VGGISH_CNT_TRAINABLE:]  # FCNs
                # print(tf.compat.v1.trainable_variables())
                train_op1 = tf.compat.v1.train.AdamOptimizer(learning_rate=lr1, epsilon=params.ADAM_EPSILON).minimize(
                    loss, var_list=var1, global_step=global_step, name="train_op1"
                )
                train_op2 = tf.compat.v1.train.AdamOptimizer(learning_rate=lr2, epsilon=params.ADAM_EPSILON).minimize(
                    loss, var_list=var2, global_step=global_step, name="train_op2"
                )  # fixed 'var1'

                train_op = tf.group(train_op1, train_op2, name="train_op")  # noqa E266
            else:
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr2, epsilon=params.ADAM_EPSILON)
                optimizer.minimize(loss, global_step=global_step, name="train_op")
        return graph


def _check_vggish_ckpt_exists():
    """check VGGish checkpoint exists or not."""
    util.maybe_create_directory(params.VGGISH_CHECKPOINT_DIR)
    if not util.is_exists(params.VGGISH_CHECKPOINT):
        raise FileNotFoundError


def break_training():
    # Break training if necessary
    if os.path.exists("stop.flag"):
        print("\nIteration interrupted on request. Model:")
        print(name_all)
        while True:
            response = input("Do you want to break training? (y/n): ")
            if response == "y":
                return True
            elif response == "n":
                os.remove("stop.flag")
                return False
    return False


# TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")


def main(_):
    if break_training():
        return

    # initialize all log data containers:
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    # test_loss_per_epoch = []
    if FLAGS["early_stop"] == "LOSS":
        val_best = 100  # loss
    elif FLAGS["early_stop"] == "AUC":
        val_best = 0  # AUC
    sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(graph=_add_triaining_graph(), config=sess_config) as sess:

        # op and tensors
        vgg_tensor = sess.graph.get_tensor_by_name(params.VGGISH_INPUT_TENSOR_NAME)
        dropout_tensor = sess.graph.get_tensor_by_name("mymodel/dropout_rate:0")
        logit_tensor = sess.graph.get_tensor_by_name("mymodel/Output/prediction:0")
        labels_tensor = sess.graph.get_tensor_by_name("train/labels:0")
        global_step_tensor = sess.graph.get_tensor_by_name("train/global_step:0")
        lr1_tensor = sess.graph.get_tensor_by_name("train/train_lr1:0")
        lr2_tensor = sess.graph.get_tensor_by_name("train/train_lr2:0")
        loss_tensor = sess.graph.get_tensor_by_name("train/loss_op:0")
        cla_loss_tensor = sess.graph.get_tensor_by_name("train/cla_loss:0")
        reg_loss_tensor = sess.graph.get_tensor_by_name("train/reg_loss2:0")
        train_op = sess.graph.get_operation_by_name("train/train_op")

        saver = tf.compat.v1.train.Saver()

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        if FLAGS["enable_checkpoint"]:
            load_vggish_slim_checkpoint(sess, params.VGGISH_CHECKPOINT)

        print(FLAGS)
        model_summary()

        checkpoint_path = os.path.join(audio_ckpt_dir, name_all + params.AUDIO_CHECKPOINT_NAME)
        if FLAGS["restore_if_possible"]:
            if FLAGS["epoch_to_restore"]:
                # saver.save(sess, f"{checkpoint_path}EPOCH{epoch}.ckpt")
                checkpoint_path_restore = f"{checkpoint_path}EPOCH{FLAGS['epoch_to_restore']}"
                if util.is_exists(checkpoint_path_restore + ".ckpt.meta"):
                    saver.restore(sess, checkpoint_path_restore)
            else:
                if util.is_exists(checkpoint_path + ".ckpt.meta"):
                    checkpoint_path_restore = f"{checkpoint_path}.ckpt"
                    saver.restore(sess, checkpoint_path_restore)

        # begin to train
        train_data, valid_data, test_data = _create_data()
        rng = np.random.default_rng()

        logfile = open(os.path.join(audio_ckpt_dir, name_all + "_log.txt"), "w")
        logfile.write("INIT testing results:")
        logfile.write("\n")

        # training and validation loop
        for epoch in range(FLAGS["epoch"]):
            # Break training if necessary
            if break_training():
                break

            # Shuffle train_data in each epoch
            rng.shuffle(train_data, axis=0)
            if epoch == 0:
                curr_step = 0

            print("--------------------------------------")
            # training loop
            train_batch_losses = []
            probs_all = []
            label_all = []
            loss_all = []
            regloss_all = []
            print("training samples:", len(train_data))

            for sample in train_data:  # generate training batch
                vgg_b, labels = util.get_input(sample)
                [num_steps, lr1, lr2, logits, loss, _, clal, regl] = sess.run(
                    [
                        global_step_tensor,
                        lr1_tensor,
                        lr2_tensor,
                        logit_tensor,
                        loss_tensor,
                        train_op,
                        cla_loss_tensor,
                        reg_loss_tensor,
                    ],
                    feed_dict={
                        vgg_tensor: vgg_b,  # Mel-spetrugram
                        dropout_tensor: [[FLAGS["dropout_rate"]]],  # traning dropour rate
                        labels_tensor: [labels],
                    },
                )  # groud truth
                probs_all.append(logits)
                label_all.append(labels[1])
                train_batch_losses.append(loss)
                loss_all.append(clal)
                regloss_all.append(regl)

            if FLAGS["is_diff"]:
                print("LEARNING RATE1:", lr1, "Learning RATE2:", lr2)
            else:
                print("LEARNING RATE:", lr2)
            # compute the train epoch loss:
            train_epoch_loss = np.mean(train_batch_losses)
            epoch_loss = np.mean(loss_all)
            epcoh_reg_loss = np.mean(regloss_all)
            # save the train epoch loss:
            train_loss_per_epoch.append(train_epoch_loss)
            print(
                "Epoch {}/{}:".format(epoch, FLAGS["epoch"]),
                "train epoch loss: %g" % train_epoch_loss,
                "cross-entropy loss: %g" % epoch_loss,
                "regulation loss: %g" % epcoh_reg_loss,
            )
            f1_score, train_AUC, train_TPR, train_TNR, train_TPR_TNR_9, accuracy = util.get_metrics(probs_all,
                                                                                                    label_all)

            logfile.write(
                "Training - Epoch {}/{}: AUC:{}, TPR:{}, TNR:{}, TPR_TNR_9:{}, Loss:{}, f1-score: {}, accuracy: {}"
                .format(
                    epoch, FLAGS["epoch"], train_AUC, train_TPR, train_TNR, train_TPR_TNR_9, train_epoch_loss, f1_score,
                    accuracy
                )
            )
            logfile.write("\n")

            # validation loop
            val_batch_losses = []
            probs_all = []
            label_all = []
            loss_all = []
            regloss_all = []
            for sample in valid_data:
                vgg_b, labels = util.get_input(sample)
                [logits, loss, clal, regl] = sess.run(
                    [logit_tensor, loss_tensor, cla_loss_tensor, reg_loss_tensor],
                    feed_dict={
                        vgg_tensor: vgg_b,
                        dropout_tensor: [[1.0]],
                        labels_tensor: [labels],
                    },
                )
                val_batch_losses.append(loss)
                probs_all.append(logits)
                label_all.append(labels[1])
                loss_all.append(clal)
                regloss_all.append(regl)

            val_loss = np.mean(val_batch_losses)
            val_loss_per_epoch.append(val_loss)
            epoch_loss = np.mean(loss_all)
            epcoh_reg_loss = np.mean(regloss_all)
            print(
                "Epoch {}/{}:".format(epoch, FLAGS["epoch"]),
                "validation loss: %g" % val_loss,
                "cross-entropy loss: %g" % epoch_loss,
                "regulation loss: %g" % epcoh_reg_loss,
            )
            f1_score, valid_AUC, valid_TPR, valid_TNR, valid_TPR_TNR_9, accuracy = util.get_metrics(probs_all,
                                                                                                    label_all)

            if valid_AUC > val_best:
                val_best = valid_AUC

            logfile.write(
                "Validation - Epoch {}/{}: AUC:{}, TPR:{}, TNR:{}, TPR_TNR_9:{}, Loss:{}, f1-score: {}, accuracy: {}"
                .format(
                    epoch, FLAGS["epoch"], valid_AUC, valid_TPR, valid_TNR, valid_TPR_TNR_9, val_loss, f1_score,
                    accuracy
                )
            )
            logfile.write("\n")

            # saver.save(sess, checkpoint_path_epoch)

            # if FLAGS["early_stop"] == "LOSS":
            #     if val_loss <= val_best:
            #         # save the model weights to disk:
            #         saver.save(sess, checkpoint_path)
            #         print("checkpoint saved in file: %s" % checkpoint_path)
            #         curr_step = 0
            #         val_best = val_loss
            #     else:
            #         curr_step += 1
            #         if curr_step == params.PATIENCE:
            #             print("Early Sopp!(Train)")
            #             logfile.write("Min Val Loss, checkpoint stored!\n")
            #             break
            #
            # elif FLAGS["early_stop"] == "AUC":
            #     val_curr = 0.5 * (valid_TPR + valid_TNR)
            #
            #     if val_best <= val_curr:
            #         if valid_TPR > 0.5 and valid_TNR > 0.5:
            #             curr_step = 0
            #             # Save the model weights to disk:
            #             saver.save(sess, checkpoint_path)
            #             print("checkpoint saved in file: %s" % checkpoint_path)
            #             val_best = val_curr
            #         else:
            #             curr_step += 1
            #             if val_best < val_curr:
            #                 curr_step = 0
            #                 val_best = val_curr
            #     else:
            #         curr_step += 1
            #
            #     if curr_step == params.PATIENCE:
            #         print("Early Stop! (Train)")
            #         logfile.write("Max Val AUC, checkpoint stored!\n")
            #         break

            # if epoch == 5:
            # print('start fine tune!')
            # train_data = train_data + valid_data

            # test loop
            if epoch == FLAGS["epoch_to_test"]:
                test_batch_losses = []
                probs_all = []
                label_all = []
                for sample in test_data:
                    vgg_b, labels = util.get_input(sample)
                    [logits, loss, _, _] = sess.run(
                        [logit_tensor, loss_tensor, cla_loss_tensor, reg_loss_tensor],
                        feed_dict={
                            vgg_tensor: vgg_b,
                            dropout_tensor: [[1.0]],
                            labels_tensor: [labels],
                        },
                    )
                    test_batch_losses.append(loss)
                    probs_all.append(logits)
                    label_all.append(labels[1])

                f1_score, test_AUC, test_TPR, test_TNR, test_TPR_TNR_9, accuracy = util.get_metrics(probs_all,
                                                                                                    label_all)

                logfile.write(
                    "Test - Epoch {}/{}: AUC:{}, TPR:{}, TNR:{}, TPR_TNR_9:{}, f1-score: {}, accuracy: {}"
                    .format(
                        epoch, FLAGS["epoch"], test_AUC, test_TPR, test_TNR, test_TPR_TNR_9, f1_score, accuracy
                    )
                )
                logfile.write("\n")

                label = util.squezze(label_all)
                probs = util.squezze(probs_all)
                predicted = util.get_predicted(probs, where_0=0)
                predicted = util.squezze(predicted)
                TN, FP, FN, TP = metrics.confusion_matrix(label, predicted).ravel()
                fpr_all, tpr_all, thresholds = metrics.roc_curve(label, probs[:, 1])
                logfile_matrix = open(f"{audio_ckpt_dir}/vgg_matrix_confusion.txt", "w")
                logfile_matrix.write(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")
                logfile_matrix.close()
                logfile_roc = open(f"{audio_ckpt_dir}/vgg_roc_curve.txt", "w")
                roc = "\n".join(
                    [f'{fpr};{tpr};{threshold}' for fpr, tpr, threshold in zip(fpr_all, tpr_all, thresholds)])
                logfile_roc.write("fpr; tpr; threshold\n")
                logfile_roc.write(roc)
                logfile_roc.close()

                # sns.set(style="whitegrid")
                # # ROC plot
                # plt.figure(figsize=(8, 8))
                # plt.plot(fpr_all, tpr_all, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(test_AUC))
                # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
                # plt.scatter(fpr_all, tpr_all, c=thresholds, cmap='viridis', label='Threshold', s=100,
                #             edgecolors='black')
                # plt.colorbar(label='Threshold')
                # plt.xlabel('False Positive Rate (FPR)')
                # plt.ylabel('True Positive Rate (TPR)')
                # plt.title('Receiver Operating Characteristic (ROC) Curve')
                # plt.legend(loc='lower right')
                # plt.show()

                # Save the model weights to disk:
                checkpoint_file_path = f"{checkpoint_path}EPOCH{epoch}.ckpt"
                saver.save(sess, checkpoint_file_path)
                print(f"checkpoint saved in file: {checkpoint_file_path} \n on Epoch: {epoch}")
                # val_best = val_curr

        logfile.write("\n")
        # nni.report_final_result(val_best)
        logfile.write("\nVal best: {}".format(val_best))
        logfile_results = open(os.path.join(audio_ckpt_dir, "results_log.txt"), "a")
        logfile_results.write("{};{}".format(name_all, val_best))
        logfile_results.write("\n")


if __name__ == "__main__":
    tf.compat.v1.app.run()

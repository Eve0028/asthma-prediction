import logging.config
import os

import numpy as np
import tensorflow as tf
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import prediction.model.model_util as util

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('training')

# Define metrics
train_loss = tf.keras.metrics.CategoricalCrossentropy('train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
# train_auc = tf.keras.metrics.AUC('train_AUC')
# train_precision = tf.keras.metrics.Precision('train_precision')
# train_recall = tf.keras.metrics.Precision('train_recall')

test_loss = tf.keras.metrics.CategoricalCrossentropy('test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')
# test_auc = tf.keras.metrics.AUC('test_AUC')
# test_precision = tf.keras.metrics.Precision('test_precision')
# test_recall = tf.keras.metrics.Precision('test_recall')


def break_training(model, parameters=None):
    # Break training if necessary
    if os.path.exists("stop.flag"):
        print("\nIteration interrupted on request. Model:")
        print(model.summary())
        if parameters:
            print(parameters)
        while True:
            response = input("Do you want to break training? (y/n): ")
            if response == "y":
                return True
            elif response == "n":
                os.remove("stop.flag")
                return False
    return False


def calculate_metrics(y_true, y_score):
    # y_true -> [[0, 1], [0, 1], ...]
    # y_score -> [[0.45, 0.55], ...]
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = util.get_predicted(y_score, where_0=0)
    tn, fp, fn, tp = confusion_matrix(y_true[:, 1], y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    auc = roc_auc_score(y_true[:, 1], y_score[:, 1])
    return tpr, tnr, f1, auc


@tf.function(reduce_retracing=True)
def train_step(model, x_train, y_train):
    with tf.GradientTape() as tape:
        scores = model(x_train, training=True)
        y_train_one = tf.convert_to_tensor(y_train[0], np.int32)
        y_train_one = tf.reshape(y_train_one, (1, 2))
        loss = model.compiled_loss(y_train_one, scores)
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(y_train_one, scores)
    train_accuracy(y_train_one, scores)
    scores = tf.reshape(scores, (2))
    return loss, scores


def test_step(model, x_test, y_test):
    scores = model(x_test)
    y_test_one = tf.convert_to_tensor(y_test[0], np.int32)
    y_test_one = tf.reshape(y_test_one, (1, 2))
    loss = model.compiled_loss(y_test_one, scores)

    test_loss(y_test_one, scores)
    test_accuracy(y_test_one, scores)
    scores = tf.reshape(scores, (2))
    return loss, scores


def evaluate_model(model, data, training=False, metrics_file=None, epoch=None, summary_writer=None):
    all_scores = []
    all_labels = []
    all_loss = []
    for sample in data:
        # sample - contains multiple images of one spectrogram and one one-hot label
        if np.shape(sample[0]) == (0, 3):  # If there is no images in one sound sample :c
            continue
        labels = np.array(sample[1])
        sample_data = sample[0]
        repeated_labels = np.repeat([labels], len(sample[0]), axis=0)

        if training:
            loss, scores = train_step(model, sample_data, repeated_labels)
        else:
            loss, scores = test_step(model, sample_data, repeated_labels)

        all_scores.append(scores)
        all_labels.append(labels)
        all_loss.append(loss)

    tpr, tnr, f1, auc = calculate_metrics(all_labels, all_scores)
    sample_loss = sum(all_loss) / len(all_loss)
    accuracy = train_accuracy.result() if training else test_accuracy.result()
    loss = train_loss.result() if training else test_loss.result()

    if summary_writer:
        with summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=epoch)
            tf.summary.scalar('accuracy', accuracy, step=epoch)
            tf.summary.scalar('TPR', tpr, step=epoch)
            tf.summary.scalar('TNR', tnr, step=epoch)
            tf.summary.scalar('F1-score', f1, step=epoch)
            tf.summary.scalar('AUC', auc, step=epoch)

    metrics_info = (f"Accuracy: {accuracy}, Loss: {sample_loss:.2f}, "
                    f"TPR: {tpr:.2f}, TNR: {tnr:.2f}, f1: {f1:.2f}, AUC: {auc:.2f}")
    logger.info(metrics_info)
    if metrics_file:
        with open(metrics_file, 'a') as f:
            f.write(f"Epoch: {epoch}, "
                    f"{metrics_info}\n")

    return sample_loss, tpr, tnr, f1, auc, all_labels, all_scores


def train(model, train_data, val_data, test_data=None, epochs=100, start_epoch=0, test_epoch=None, patience=15,
          metrics_file=None, early_stopping=None, save_checkpoint_file=None, load_checkpoint_file=None, batch_size=1):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + Path(metrics_file).stem + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + Path(metrics_file).stem + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    best_val_tpr_tnr = 0.0  # 0.5 * (tpr + tnr)
    curr_step = 0
    rng = np.random.default_rng()
    for epoch in range(epochs - start_epoch):
        # Break training if necessary
        if break_training(model, save_checkpoint_file):
            return
        # Shuffle train_data in each epoch
        rng.shuffle(train_data, axis=0)
        logger.info(f"\nEpoch: {epoch + start_epoch}")
        # Training.
        logger.info("Training")
        evaluate_model(model, train_data, training=True,
                       metrics_file=metrics_file, epoch=epoch + start_epoch,
                       summary_writer=train_summary_writer)
        # Validation.
        logger.info("Validation")
        _, tpr, tnr, _, _, _, _ = evaluate_model(model, val_data, training=False,
                                                 metrics_file=metrics_file, epoch=epoch + start_epoch,
                                                 summary_writer=test_summary_writer)

        test_loss.reset_states()
        test_accuracy.reset_states()

        # Testing.
        if test_epoch is not None and test_data and test_epoch == epoch + start_epoch:
            logger.info("Testing")
            _, _, _, _, auc, all_labels, all_scores = evaluate_model(model, test_data, training=False,
                                                                     metrics_file=metrics_file,
                                                                     epoch=epoch + start_epoch)

            all_labels = np.array(all_labels)
            all_scores = np.array(all_scores)
            predicted_arr = util.get_predicted(all_scores, where_0=0)

            TN, FP, FN, TP = confusion_matrix(all_labels[:, 1], predicted_arr).ravel()
            fpr_all, tpr_all, thresholds = roc_curve(all_labels[:, 1], all_scores[:, 1])
            with open(f"{metrics_file}_matrix_confusion.txt", 'w') as f:
                f.write(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")
            with open(f"{metrics_file}_vgg_roc_curve.txt", 'w') as f:
                roc = "\n".join(
                    [f'{fpr};{tpr};{threshold}' for fpr, tpr, threshold in zip(fpr_all, tpr_all, thresholds)])
                f.write("fpr;tpr;threshold\n")
                f.write(roc)

            # Save the model weights to disk:
            logger.info("Validation TPR/TNR improved. Saving best weights.")
            model.save_weights(f"{save_checkpoint_file}EPOCH{epoch}.weights.h5")

            # sns.set(style="whitegrid")
            # # ROC plot
            # plt.figure(figsize=(8, 8))
            # plt.plot(fpr_all, tpr_all, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(auc))
            # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
            # plt.scatter(fpr_all, tpr_all, c=thresholds, cmap='viridis', label='Threshold', s=100,
            #             edgecolors='black')
            # plt.colorbar(label='Threshold')
            # plt.xlabel('False Positive Rate (FPR)')
            # plt.ylabel('True Positive Rate (TPR)')
            # plt.title('Receiver Operating Characteristic (ROC) Curve')
            # plt.legend(loc='lower right')
            # plt.show()

        # Early stopping and save model weights.
        if early_stopping:
            curr_val_tpr_tnr = 0.5 * (tpr + tnr)
            if best_val_tpr_tnr <= curr_val_tpr_tnr:
                if tpr > 0.5 and tnr > 0.5:
                    curr_step = 0
                    # Save the model weights to disk:
                    logger.info("Validation TPR/TNR improved. Saving best weights.")
                    model.save_weights(f"{save_checkpoint_file}EPOCH{epoch}.weights.h5")
                    best_val_tpr_tnr = curr_val_tpr_tnr
                else:
                    curr_step += 1
                    if best_val_tpr_tnr < curr_val_tpr_tnr:
                        curr_step = 0
                        best_val_tpr_tnr = curr_val_tpr_tnr
            else:
                curr_step += 1

            if curr_step >= patience:
                print("Early Stop! (Train)")
                break
        else:
            curr_val_tpr_tnr = 0.5 * (tpr + tnr)
            if best_val_tpr_tnr < curr_val_tpr_tnr:
                best_val_tpr_tnr = curr_val_tpr_tnr

        # Reset metrics every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    # Save best results to file.
    logger.info(f"Best val: {best_val_tpr_tnr}\n")
    if metrics_file:
        with open(metrics_file, 'a') as f:
            f.write(f"Best val: {best_val_tpr_tnr}\n")

    return model

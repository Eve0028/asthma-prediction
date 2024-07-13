import tensorflow as tf

from load_data import load_data
from model import get_model
from params import early_stop_patience, initial_epochs, number_of_finetune_layers, max_epochs, signals_data_dir, \
    batch_size
from preprocessing import get_steps, prepare_dataset


# Custom combined TPR and TNR metric
class CombinedTPRTNR(tf.keras.metrics.Metric):
    def __init__(self, name='combined_tpr_tnr', **kwargs):
        super(CombinedTPRTNR, self).__init__(name=name, **kwargs)
        self.true_positives = tf.keras.metrics.TruePositives()
        self.true_negatives = tf.keras.metrics.TrueNegatives()
        self.false_positives = tf.keras.metrics.FalsePositives()
        self.false_negatives = tf.keras.metrics.FalseNegatives()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self):
        tpr = self.true_positives.result() / (
                self.true_positives.result() + self.false_negatives.result() + tf.keras.backend.epsilon())
        tnr = self.true_negatives.result() / (
                self.true_negatives.result() + self.false_positives.result() + tf.keras.backend.epsilon())
        return 0.5 * (tpr + tnr)

    def reset_states(self):
        self.true_positives.reset_states()
        self.true_negatives.reset_states()
        self.false_positives.reset_states()
        self.false_negatives.reset_states()


# Add Early Stopping based on combined TPR and TNR
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_combined_tpr_tnr', patience=early_stop_patience,
                                                  mode='max', verbose=1, restore_best_weights=True)

time_distributed_vggish = get_model()

train, val, test = load_data(signals_data_dir, samples_count=None, ratios=(0.7, 0.2, 0.1))
train_ds = prepare_dataset(train, batch_size, padded=False, shuffle=True)
val_ds = prepare_dataset(val, batch_size, padded=False)
test_ds = prepare_dataset(test, batch_size, padded=False, repeat=False)

steps_per_epoch_train = get_steps(train)
steps_per_epoch_val = get_steps(val)
steps_per_epoch_test = get_steps(test)

# Freeze the base model except the last 4 TimeDistributed layers and beyond (+2 for pooling and softmax at the end)
for layer in time_distributed_vggish.layers[:-6]:
    layer.trainable = False

# Compile the model with the frozen layers
time_distributed_vggish.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', CombinedTPRTNR()]
)

# Train only the new layers for a few epochs (convergence)
history = time_distributed_vggish.fit(
    train_ds,
    epochs=initial_epochs,
    steps_per_epoch=steps_per_epoch_train,
    validation_data=val_ds,
    validation_steps=steps_per_epoch_val,
    callbacks=[early_stopping]
)

# Unfreeze specified number of layers
# (plus the last 4 TimeDistributed layers +2 for pooling and softmax at the end) for fine-tuning
for layer in time_distributed_vggish.layers[-(number_of_finetune_layers + 6):]:
    layer.trainable = True

# Compile the model with the unfrozen layers
time_distributed_vggish.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate for fine-tuning
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', CombinedTPRTNR()]
)

# Fine-tune the rest
fine_tune_epochs = max_epochs - initial_epochs
history_fine = time_distributed_vggish.fit(
    train_ds,
    epochs=fine_tune_epochs,
    steps_per_epoch=steps_per_epoch_train,
    validation_data=val_ds,
    validation_steps=steps_per_epoch_val,
    callbacks=[early_stopping]
)

test_loss, test_accuracy, test_combined_tpr_tnr = time_distributed_vggish.evaluate(test_ds)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Combined TPR/TNR: {test_combined_tpr_tnr}")

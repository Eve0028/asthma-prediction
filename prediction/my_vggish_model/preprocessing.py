import tensorflow as tf

from params import sample_rate, batch_size, sample_width, sample_height
from prediction.vggish.vggish_input import waveform_to_examples


def preprocess_audio(breath):
    examples = waveform_to_examples(breath, sample_rate)
    return examples


def preprocess_sample(sample):
    breath, label = sample['breath'], sample['label']
    frames = preprocess_audio(breath)
    return frames, label


def create_dataset(data):
    def generator():
        for sample in data:
            frames, label = preprocess_sample(sample)
            yield frames, label

    output_signature = (
        tf.TensorSpec(shape=(None, 96, 64), dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.int32)
    )

    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)


def prepare_dataset(data, batch_size, padded=True, shuffle=False, repeat=True):
    dataset = create_dataset(data)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    if shuffle:
        dataset = dataset.shuffle(1000)
    if padded:
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([None, sample_width, sample_height]),
                tf.TensorShape([2])
            )
        )
    else:
        dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()

    return dataset.prefetch(buffer_size=AUTOTUNE)


def get_steps(data):
    steps_per_epoch = len(data) // batch_size
    return steps_per_epoch

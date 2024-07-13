import os
from pathlib import Path
import logging.config

import numpy as np
import skimage
import matplotlib.pyplot as plt
import librosa

from prediction.vggish import mel_features  # use VGGish spectrogram process
import prediction.my_image_model.model_params as params
import prediction.model.model_util as util

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('preprocess')

SR = 16000  # sample rate
FRAME_LEN = int(SR / 10)  # 10 ms
HOP = int(FRAME_LEN / 2)  # 50%overlap, 5ms

# Paths
signals_data_dir = os.path.join(params.TF_DATA_DIR, params.DATA_NAME)  # ./data


def show_spectrogram(spectrogram, sr=16000, path=None):
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel')
    if path:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def img_show(img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.close()


def load_singals_datasets(samples_count=None, ratios=(0.7, 0.2, 0.1)):
    """Create audio `train`, `test` and `val` records file."""
    logger.info("Create records..")
    train, val, test = util.load_data(signals_data_dir, samples_count=samples_count, ratios=ratios)
    logger.info("Dataset size: Train-{} Test-{} Val-{}".format(len(train), len(test), len(val)))
    return train, val, test


def vggish_log_mel_spectrogram(data, sr=16000, log_offset=0.01, n_fft_sec=0.025, hop_sec=0.01,
                               n_mels=64, fmin=125, fmax=7500):
    return mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=sr,
        log_offset=log_offset,
        window_length_secs=n_fft_sec,
        hop_length_secs=hop_sec,
        num_mel_bins=n_mels,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
    )


def scale_minmax(X, min, max):
    std = (X - X.min()) / (X.max() - X.min())
    X_scaled = std * (max - min) + min
    return X_scaled


def spectrogram_to_image(spectrogram, path=None, show=False, flip=True):
    img = scale_minmax(spectrogram, 0, 255).astype(np.uint8)
    if flip:
        img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy
    if path:
        skimage.io.imsave(path, img)
    if show:
        img_show(img)
    return img


def spectrogram_to_images(spectrogram, out=None, window_length=224, hop_length=None):
    if hop_length is None:
        hop_length = window_length
    if out:
        path = Path(out)
        if not path.name.endswith('.png'):
            raise Exception(f"{out} does not end with .png")
    else:
        path = None

    if path:
        spec_whole_image = spectrogram_to_image(spectrogram, show=False, flip=False)
    else:
        spec_whole_image = spectrogram_to_image(spectrogram, show=False, flip=False)
    spec_images = mel_features.frame(spec_whole_image, window_length, hop_length)
    spec_images = [np.flip(spec_img.T, axis=0) for spec_img in spec_images]
    return spec_images


def signal_to_images(signal, out=None, window_length=224, hop_length=None, n_mels=128):
    vggish_log_mel_spec = vggish_log_mel_spectrogram(signal, n_mels=n_mels)
    return spectrogram_to_images(vggish_log_mel_spec, out=out,
                                 window_length=window_length, hop_length=hop_length)


def image_resize(img, shape):
    new_height, new_width = shape
    # Interpolation
    resized_img = skimage.transform.resize(img, (new_height, new_width))
    # Scale to [0, 255]
    rescaled_img = skimage.img_as_ubyte(resized_img)
    return rescaled_img


def gen_images_dataset(data, preprocess=None, shape=(224, 224), n_mels=128):
    """
    Args:
        preprocess: None
        data: [{"breath": [signal], "label": [labels]}]
        shape: shape of dataset images
        n_mels: number of mel filters

    Returns: batches (samples) of 3-channel images [[[images, ...], labels], ]
    """
    input_data = []
    height, width = shape
    for sample in data:
        x = signal_to_images(sample["breath"],
                             window_length=width, n_mels=n_mels)  # get images from one sound sample
        if height > n_mels:
            for idx, img in enumerate(x):
                x[idx] = image_resize(img, shape)

        x = np.repeat(np.array(x)[..., np.newaxis], 3, -1)  # grayscale to rgb

        if preprocess:
            x = preprocess(x)

        input_data.append([x, sample["label"]])
    return input_data

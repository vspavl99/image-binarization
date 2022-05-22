import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from typing import Tuple
import bottleneck as bn
from src.utils import check_image_channels


def niblack(image: np.ndarray, window_size: tuple = (25, 25), k: float = -0.2):
    """
    Appl niblack thresholding for given image
    :param image: image in grayscale
    :param window_size: size of sliding windows
    :param k: hyperparameter
    :return: binarized image
    """

    if not check_image_channels(image):
        print('Image must be in gray scale!')
        return None

    image = image.astype(np.float64)

    mean_kernel = np.ones((window_size[0], window_size[1])) / (window_size[0] * window_size[1])

    means = scipy.signal.convolve2d(image, mean_kernel, mode='same', boundary='symm')
    deviations = np.sqrt(
        scipy.signal.convolve2d(image ** 2, mean_kernel, mode='same', boundary='symm') - means ** 2
    )

    threshold_py_pixel = means + k * deviations

    # padded = np.pad(
    #     image, ((window_size[0] // 2, window_size[0] // 2), (window_size[1] // 2, window_size[1] // 2)), mode='reflect'
    # ).astype(np.float32)

    # windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_size)
    # means = np.mean(windows, axis=(2, 3), dtype=np.float64)
    # deviations = np.std(windows, axis=(2, 3), dtype=np.float64, dtype=np.float32)
    # threshold_py_pixel = means + k * deviations

    return threshold_py_pixel

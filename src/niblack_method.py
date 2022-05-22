from typing import Union

import numpy as np
import scipy.signal

from src.utils import check_image_channels, sliding_windows


def niblack(image: np.ndarray, window_size: tuple = (25, 25), k: float = -0.2) -> Union[np.ndarray, None]:
    """
    Apply niblack thresholding for given image
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
        np.abs(scipy.signal.convolve2d(image ** 2, mean_kernel, mode='same', boundary='symm') - means ** 2)
    )

    threshold_py_pixel = means + k * deviations

    return image > threshold_py_pixel


def modified_niblack(image: np.ndarray, window_size: tuple = (31, 31), k: float = -0.2, min_deviation: float = 10)\
        -> Union[np.ndarray, None]:
    """
    Apply modified niblack thresholding for given image
    :param image: image in grayscale
    :param window_size: size of sliding windows
    :param k: hyperparameter
    :param min_deviation: hyperparameter
    :return: binarized image
    """

    if not check_image_channels(image):
        print('Image must be in gray scale!')
        return None

    image = image.astype(np.float64)
    padded = np.pad(
        image, ((window_size[0] // 2, window_size[0] // 2), (window_size[1] // 2, window_size[1] // 2)), mode='reflect'
    ).astype(np.float32)

    thresholds = np.empty_like(image)

    windows = sliding_windows(padded.shape, window_size=window_size)

    for center, top_left_x, top_left_y, bottom_right_x, bottom_right_y in windows:

        is_threshold_find = False

        while not is_threshold_find:
            
            sub_image = padded[top_left_x:bottom_right_x, top_left_y:bottom_right_y]
            mean = np.mean(sub_image)
            deviation = np.std(sub_image)

            if deviation > min_deviation:
                threshold_py_pixel = mean + k * deviation

                thresholds[center[0], center[1]] = threshold_py_pixel
                is_threshold_find = True

            else:
                top_left_x = max(top_left_x - window_size[0], 0)
                top_left_y = max(top_left_y - window_size[1], 0)

                bottom_right_x = min(bottom_right_x + window_size[0], padded.shape[0])
                bottom_right_y = min(bottom_right_y + window_size[1], padded.shape[1])

                if bottom_right_x == padded.shape[0] or bottom_right_x == padded.shape[1]:
                    print('Out of border')
                    return image

    return image > thresholds



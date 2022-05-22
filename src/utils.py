import numpy as np


def check_image_channels(image: np.ndarray) -> bool:
    """Check that image is grayscale"""

    if len(image.shape) == 2 or image.shape[2] == 1:
        return True

    return False


def sliding_windows(image_size, window_size=(3, 3)):

    width, height = image_size

    for i in range(width - window_size[0]):

        for j in range(height - window_size[1]):

            center_point_on_original = i, j
            top_left_x, top_left_y = center_point_on_original
            botton_right_x, botton_right_y = i + window_size[0], j + window_size[1]

            yield center_point_on_original, top_left_x, top_left_y, botton_right_x, botton_right_y

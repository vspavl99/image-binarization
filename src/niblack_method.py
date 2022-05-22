import numpy as np

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

    binarized_image = np.zeros_like(image)
    image = np.pad(image, ((window_size[0] // 2, window_size[0] // 2), (window_size[1] // 2, window_size[1] // 2)),
                   mode='reflect')
    sum_images = np.lib.stride_tricks.sliding_window_view(image, window_shape=window_size)

    for i, _ in enumerate(sum_images):
        for j, window in enumerate(_):
            mean = np.mean(window)
            variance = np.sqrt(np.var(window))
            threshold = mean + k * variance

            relative_coord = window_size[0] // 2, window_size[1] // 2
            absolute_coord = i, j

            thresholded_pixel = threshold
            binarized_image[absolute_coord[0]][absolute_coord[1]] = thresholded_pixel

    return binarized_image

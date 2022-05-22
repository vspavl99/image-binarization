from typing import Union, Tuple

import cv2
import numpy as np

from src.utils import check_image_channels


def compute_statistics(normalised_distribution: np.ndarray, level: int):
    all_levels = np.arange(start=0, stop=len(normalised_distribution))

    background_levels, object_levels = all_levels[:level], all_levels[level:]
    background_probabilities, object_probabilities = normalised_distribution[:level], normalised_distribution[level:]

    cumulative_sum = normalised_distribution.cumsum()
    background_probability_sum = cumulative_sum[level]
    object_probability_sum = cumulative_sum[255] - cumulative_sum[level]

    if background_probability_sum < 1.e-6 or object_probability_sum < 1.e-6:
        return dict()

    background_expected_value = np.sum(background_levels * background_probabilities) / background_probability_sum
    overall_expected_value = (all_levels * normalised_distribution).sum()
    object_expected_value = \
        (overall_expected_value - background_expected_value * background_probability_sum) / object_probability_sum

    statistics = {
        'background_levels': background_levels,
        'object_levels': object_levels,
        'background_probabilities': background_probabilities,
        'object_probabilities': object_probabilities,
        'background_probability_sum': background_probability_sum,
        'object_probability_sum': object_probability_sum,
        'background_expected_value': background_expected_value,
        'object_expected_value': object_expected_value,
        'overall_expected_value': overall_expected_value
    }

    return statistics


def compute_otsu_criteria(stats: dict) -> float:
    """Compute optimization criteria at certain level"""
    if not stats:
        return np.infty

    background_variance = np.sum(
        stats['background_probabilities'] * (stats['background_levels'] - stats['background_expected_value']) ** 2
    ) / stats['background_probability_sum']

    object_variance = np.sum(
        stats['object_probabilities'] * (stats['object_levels'] - stats['object_expected_value']) ** 2
    ) / stats['object_probability_sum']

    withinclass_variance = stats['background_probability_sum'] * background_variance +\
        stats['object_probability_sum'] * object_variance

    return withinclass_variance


def compute_modified_otsu_criteria(stats: dict) -> float:
    if not stats:
        return -np.infty

    withinclass_variance = compute_otsu_criteria(stats)

    criteria = stats['background_probability_sum'] * np.log(stats['background_probability_sum']) +\
        stats['object_probability_sum'] * np.log(stats['object_probability_sum']) -\
        np.log(withinclass_variance)

    return criteria


def otsu(image: np.ndarray, type_of_method='original') -> Union[Tuple[np.ndarray, int], None]:
    """
    Apply otsu-method for binarization.
    :param image: image in grayscale for binarization
    :param type_of_method: 'original' or 'modified'
    :return: Binarized image with threshold or None
    """

    if not check_image_channels(image):
        print('Image must be in gray scale!')
        return None

    if type_of_method == 'original':
        compute_criteria = compute_otsu_criteria
        optimal_selection = np.argmin
    elif type_of_method == 'modified':
        compute_criteria = compute_modified_otsu_criteria
        optimal_selection = np.argmax
    else:
        print('Unknown type of method')
        return None

    values_distribution = cv2.calcHist([image], [0], None, [256], [0, 256])
    values_distribution = values_distribution.ravel()

    number_of_pixels = image.size

    assert number_of_pixels == int(values_distribution.sum())

    normalised_values_distribution = values_distribution / number_of_pixels

    criterias = []
    for level in range(1, 256):
        statistics_at_current_level = compute_statistics(normalised_values_distribution, level)
        criterias.append(compute_criteria(statistics_at_current_level))

    optimal_level = optimal_selection(np.array(criterias))

    return image >= optimal_level, int(optimal_level)


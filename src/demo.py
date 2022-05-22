import os
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.otsu_method import otsu
from src.niblack_method import niblack, modified_niblack


def compare_otsu():
    for i, file_name_with_extension in enumerate(os.listdir('data/raw/')):

        image = cv2.imread(f'data/raw/{file_name_with_extension}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        binarized_image_otsu, threshold_otsu = otsu(image, type_of_method='original')
        binarized_image_modified_otsu, threshold_modified_otsu = otsu(image, type_of_method='modified')

        file_name = file_name_with_extension.split('.')[0]
        cv2.imwrite(f'data/otsu/{file_name}_thr_{threshold_otsu}.jpg', binarized_image_otsu.astype(np.uint8) * 255)

        cv2.imwrite(
            f'data/otsu_modified/{file_name}_thr_{threshold_modified_otsu}.jpg',
            binarized_image_modified_otsu.astype(np.uint8) * 255
        )

        pixels = image.ravel()
        (unique, counts) = np.unique(pixels, return_counts=True)

        plt.title(file_name)
        plt.hist(pixels, 255)

        plt.vlines(threshold_otsu, 0, counts.max(), color='m', label='otsu')
        plt.vlines(threshold_modified_otsu, 0, counts.max(), color='c', label='modified otsu')
        plt.xlim([0, 255])

        plt.legend()
        plt.savefig(f'data/histograms/{file_name}.jpg')

        plt.show()


def compute_niblack():
    for i, file_name_with_extension in enumerate(os.listdir('data/raw/')):

        image = cv2.imread(f'data/raw/{file_name_with_extension}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        binarized_image_niblack = niblack(image, window_size=(51, 51), k=-0.5)
        binarized_image_modified_niblack, _ = modified_niblack(image, window_size=(61, 61), k=-0.2, min_deviation=11)

        file_name = file_name_with_extension.split('.')[0]
        cv2.imwrite(f'data/niblack/{file_name}.jpg', binarized_image_niblack.astype(np.uint8) * 255)

        cv2.imwrite(
            f'data/niblack_modified/{file_name}.jpg',
            binarized_image_modified_niblack.astype(np.uint8)
        )


if __name__ == "__main__":
    # compute_niblack()
    # compare_otsu()

    image = cv2.imread(f'data/raw/15.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized_image_niblack = niblack(image, window_size=(201, 201), k=-0.5)
    plt.imshow(binarized_image_niblack, cmap='gray')
    plt.show()

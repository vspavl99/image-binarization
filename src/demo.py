import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.otsu_method import otsu
from src.utils import otsu_opencv, otsu_skimage


def compare_otsu():
    for i, file_name in enumerate(os.listdir('data/raw/')):

        if file_name != '6.png':
            continue

        test_image = cv2.imread(f'data/raw/{file_name}')
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        binarized_image, th1 = otsu(test_image)

        # Otsu's thresholding
        th2, cv2_binarized_image = cv2.threshold(test_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        th3 = otsu_opencv(test_image)
        th4 = otsu_skimage(test_image)

        print(th1, th2, th3, th4)

        plt.subplot(2, 2, 1)
        plt.imshow(test_image, cmap='gray')

        plt.subplot(2, 2, 2)
        plt.imshow(binarized_image, cmap='gray')

        plt.subplot(2, 2, 3)
        plt.imshow(binarized_image, cmap='gray')

        plt.subplot(2, 2, 4)
        plt.imshow(cv2_binarized_image, cmap='gray')
        plt.show()


if __name__ == "__main__":
    compare_otsu()

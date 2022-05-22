import os
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_niblack

from src.otsu_method import otsu
from src.niblack_method import niblack
from src.utils import otsu_opencv, otsu_skimage


def compare_otsu():
    for i, file_name in enumerate(os.listdir('data/raw/')):
        print(file_name)
        test_image = cv2.imread(f'data/raw/{file_name}')
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        binarized_image, th1 = otsu(test_image, type_of_method='original')
        binarized_image2, th1_1 = otsu(test_image, type_of_method='modified')
        #
        # Otsu's thresholding
        th2, cv2_binarized_image = cv2.threshold(test_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        th3 = otsu_opencv(test_image)
        th4 = otsu_skimage(test_image)
        #
        print(f'My realisation otsu {th1}. Modified realisation {th1_1}. Opencv {th2}. Opencv example {th3}. Skimage {th4}')


        # plt.subplot(2, 2, 1)
        # plt.imshow(test_image, cmap='gray')
        #
        # plt.subplot(2, 2, 2)
        # plt.imshow(binarized_image, cmap='gray')
        #
        # plt.subplot(2, 2, 3)
        # plt.imshow(binarized_image2, cmap='gray')
        #
        # plt.subplot(2, 2, 4)
        # plt.imshow(cv2_binarized_image, cmap='gray')
        # plt.show()
        # pixels = test_image.ravel()
        # plt.title(file_name)
        # plt.hist(pixels, 255)
        # plt.xlim([0, 255])
        # plt.show()

def compute_niblack():
    for i, file_name in enumerate(os.listdir('data/raw/')):
        print(file_name)
        test_image = cv2.imread(f'data/raw/{file_name}')
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        t1 = time.time()
        thresh_niblack_custom = niblack(test_image)
        result_custom = test_image > thresh_niblack_custom
        t2 = time.time()
        thresh_niblack_skimage = threshold_niblack(test_image, window_size=25)
        result_skimage = test_image > thresh_niblack_skimage
        t3 = time.time()

        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(test_image, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title(f'Custom {t2-t1:.2f}')
        plt.imshow(result_custom, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title(f'Skimage {t3-t2:.2f}')
        plt.imshow(result_skimage, cmap='gray')
        plt.show()

        print(bool(np.sum(result_skimage != result_custom)))


if __name__ == "__main__":
    compute_niblack()

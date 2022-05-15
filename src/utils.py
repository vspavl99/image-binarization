# from typing import Bool

import cv2
import numpy as np
import matplotlib.pyplot as plt


def check_image_channels(image: np.ndarray) -> bool:
    """Check that image is grayscale"""

    if len(image.shape) == 2 or image.shape[2] == 1:
        return True

    return False

def otsu_opencv(img):

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    fn_max = 0
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2

        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        fn2 = q1*q2*(m2-m1)**2
        # print(m1, m2, fn2)
        if fn < fn_min:
            fn_min = fn
            thresh = i
        if fn2 > fn_max:
            fn_max = fn2
            thresh2 = i

    return thresh


def otsu_skimage(image, is_normalized=True):
    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.sum())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]

    # for i in range(len(mean1)):
    #     print(mean1[i], mean2[i], inter_class_variance[i])

    # print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold

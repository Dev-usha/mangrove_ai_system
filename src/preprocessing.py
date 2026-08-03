import cv2
import numpy as np


def segment_leaf(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    lower_green = np.array([20, 30, 30])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def suppress_veins(gray_leaf):
    h, w = gray_leaf.shape

    k = max(15, int(min(h, w) * 0.03))

    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    background = cv2.morphologyEx(gray_leaf, cv2.MORPH_OPEN, kernel)
    suppressed = cv2.subtract(gray_leaf, background)

    suppressed = cv2.normalize(
        suppressed,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return suppressed.astype(np.uint8)


def normalize_leaf_color(leaf_rgb):
    lab = cv2.cvtColor(leaf_rgb, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))

    result = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    return result
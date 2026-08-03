import os
import random
import cv2
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from scipy.fftpack import fft2, fftshift
import pywt

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage.filters import gabor

from src.config import DATASET_ROOT, FEATURE_CSV, SAMPLES_PER_IMAGE, PATCH_SIZE
from src.data_loader import find_images, load_image
from src.preprocessing import segment_leaf, suppress_veins, normalize_leaf_color


def add_gabor_and_wavelet(patch_gray, features):
    gab_vals = []

    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        real, imag = gabor(patch_gray, frequency=0.2, theta=theta)
        gab_vals.append(np.mean(np.abs(real)))

    features["gabor_mean"] = np.mean(gab_vals)
    features["gabor_std"] = np.std(gab_vals)

    coeffs = pywt.dwt2(patch_gray, "db1")
    cA, (cH, cV, cD) = coeffs

    features["wavelet_cH_energy"] = np.sum(cH ** 2)
    features["wavelet_cV_energy"] = np.sum(cV ** 2)
    features["wavelet_cD_energy"] = np.sum(cD ** 2)

    return features


def extract_features_at_pixel(img_rgb, img_gray_suppressed, mask, x, y, patch_size=7):
    features = {}

    half = patch_size // 2
    h, w = img_gray_suppressed.shape

    if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
        return None

    if mask[y, x] == 0:
        return None

    patch_rgb = img_rgb[y-half:y+half+1, x-half:x+half+1]
    patch_gray = img_gray_suppressed[y-half:y+half+1, x-half:x+half+1]
    patch_mask = mask[y-half:y+half+1, x-half:x+half+1]

    valid_pixels = patch_gray[patch_mask > 0]

    if len(valid_pixels) < 5:
        return None

    for i, channel_name in enumerate(["R", "G", "B"]):
        vals = patch_rgb[:, :, i][patch_mask > 0]

        features[f"{channel_name}_mean"] = np.mean(vals)
        features[f"{channel_name}_std"] = np.std(vals)

    features["gray_mean"] = np.mean(valid_pixels)
    features["gray_std"] = np.std(valid_pixels)

    if np.std(valid_pixels) > 1e-6:
        features["gray_skew"] = skew(valid_pixels)
        features["gray_kurtosis"] = kurtosis(valid_pixels)
    else:
        features["gray_skew"] = 0
        features["gray_kurtosis"] = 0

    features["entropy"] = shannon_entropy(patch_gray)

    lbp = local_binary_pattern(
        patch_gray,
        P=8,
        R=1,
        method="uniform"
    )

    features["lbp_mean"] = np.mean(lbp)

    quant = (patch_gray // 16).astype(np.uint8)

    glcm = graycomatrix(
        quant,
        distances=[1],
        angles=[0],
        levels=16,
        symmetric=True,
        normed=True
    )

    features["glcm_contrast"] = graycoprops(glcm, "contrast")[0, 0]
    features["glcm_energy"] = graycoprops(glcm, "energy")[0, 0]
    features["glcm_homogeneity"] = graycoprops(glcm, "homogeneity")[0, 0]

    fft_vals = np.abs(fftshift(fft2(patch_gray)))

    features["fft_mean"] = np.mean(fft_vals)
    features["fft_std"] = np.std(fft_vals)

    features = add_gabor_and_wavelet(patch_gray, features)

    return features


def create_feature_dataset():
    if os.path.exists(FEATURE_CSV):
        print("Feature CSV already exists.")
        print("Loading:", FEATURE_CSV)
        return pd.read_csv(FEATURE_CSV)

    image_paths = find_images(DATASET_ROOT)

    if len(image_paths) == 0:
        raise ValueError(
            f"No images found inside {DATASET_ROOT}. "
            "Check whether datamangrove is placed correctly."
        )

    print("Total images found:", len(image_paths))

    features_list = []

    for img_index, img_path in enumerate(image_paths, start=1):
        try:
            species = os.path.basename(os.path.dirname(img_path))
            image_id = os.path.basename(img_path)

            img = load_image(img_path)

            mask = segment_leaf(img)

            if np.sum(mask > 0) == 0:
                print("Skipping image with empty leaf mask:", img_path)
                continue

            leaf_rgb = cv2.bitwise_and(img, img, mask=mask)

            gray_leaf = cv2.cvtColor(leaf_rgb, cv2.COLOR_RGB2GRAY)
            gray_supp = suppress_veins(gray_leaf)

            leaf_norm = normalize_leaf_color(leaf_rgb)

            h, w = gray_supp.shape

            collected = 0
            attempts = 0
            max_attempts = SAMPLES_PER_IMAGE * 10

            while collected < SAMPLES_PER_IMAGE and attempts < max_attempts:
                attempts += 1

                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)

                f = extract_features_at_pixel(
                    leaf_norm,
                    gray_supp,
                    mask,
                    x,
                    y,
                    patch_size=PATCH_SIZE
                )

                if f is not None:
                    f["label"] = species
                    f["x"] = x
                    f["y"] = y
                    f["image_id"] = image_id

                    features_list.append(f)
                    collected += 1

            if img_index % 10 == 0:
                print(
                    f"Processed {img_index}/{len(image_paths)} images. "
                    f"Total patches: {len(features_list)}"
                )

        except Exception as e:
            print("Error processing:", img_path)
            print("Reason:", e)

    df = pd.DataFrame(features_list)

    os.makedirs(os.path.dirname(FEATURE_CSV), exist_ok=True)

    df.to_csv(FEATURE_CSV, index=False)

    print("Feature extraction completed.")
    print("Saved to:", FEATURE_CSV)
    print("Shape:", df.shape)

    return df
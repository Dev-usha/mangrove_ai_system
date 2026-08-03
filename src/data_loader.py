import os
import cv2

from src.config import DATASET_ROOT


def check_dataset_exists():
    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}\n"
            "Place your datamangrove folder inside the data folder."
        )

    class_folders = [
        folder for folder in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, folder))
    ]

    if len(class_folders) == 0:
        raise ValueError(
            f"No class folders found inside {DATASET_ROOT}.\n"
            "Expected folders like avicennia_alba, rhizophora_apiculata, sonneratia_alba."
        )

    print("Dataset found.")
    print("Classes:", class_folders)


def find_images(root, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    image_paths = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(exts):
                image_paths.append(os.path.join(dirpath, fname))

    return sorted(image_paths)


def load_image(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Could not load image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
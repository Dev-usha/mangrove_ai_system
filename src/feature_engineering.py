import numpy as np
import pandas as pd


def add_visual_health_features(df):
    """
    Add interpretable image-based vegetation and stress features.

    These features help the model understand:
    - greenness
    - brightness
    - red-green balance
    - color stress
    - texture stress

    Input:
        df: patch-level feature dataframe

    Output:
        df: dataframe with additional engineered features
    """

    df = df.copy()

    required_cols = [
        "R_mean",
        "G_mean",
        "B_mean",
        "R_std",
        "G_std",
        "B_std",
        "gray_mean",
        "gray_std",
        "entropy",
        "glcm_contrast",
        "glcm_energy",
        "glcm_homogeneity"
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns for feature engineering: {missing_cols}"
        )

    eps = 1e-6

    # =====================================================
    # Color-based vegetation features
    # =====================================================

    df["green_ratio"] = (
        df["G_mean"] /
        (df["R_mean"] + df["G_mean"] + df["B_mean"] + eps)
    )

    df["red_ratio"] = (
        df["R_mean"] /
        (df["R_mean"] + df["G_mean"] + df["B_mean"] + eps)
    )

    df["blue_ratio"] = (
        df["B_mean"] /
        (df["R_mean"] + df["G_mean"] + df["B_mean"] + eps)
    )

    df["excess_green"] = (
        2 * df["G_mean"] - df["R_mean"] - df["B_mean"]
    )

    df["red_green_diff"] = (
        df["G_mean"] - df["R_mean"]
    )

    df["green_blue_diff"] = (
        df["G_mean"] - df["B_mean"]
    )

    df["red_green_ratio"] = (
        df["R_mean"] /
        (df["G_mean"] + eps)
    )

    df["green_red_ratio"] = (
        df["G_mean"] /
        (df["R_mean"] + eps)
    )

    # =====================================================
    # Brightness and color variability features
    # =====================================================

    df["brightness_index"] = (
        df["R_mean"] + df["G_mean"] + df["B_mean"]
    ) / 3

    df["color_std_mean"] = (
        df["R_std"] + df["G_std"] + df["B_std"]
    ) / 3

    df["color_variability_index"] = (
        df["R_std"] + df["G_std"] + df["B_std"]
    ) / (
        df["brightness_index"] + eps
    )

    # =====================================================
    # Chlorosis-like visual stress indicators
    # =====================================================
    # These are not laboratory chlorophyll values.
    # They are visual indicators of greenness/discoloration.

    df["chlorosis_visual_index"] = (
        df["R_mean"] + df["B_mean"]
    ) / (
        2 * df["G_mean"] + eps
    )

    df["greenness_strength"] = (
        df["G_mean"] - np.maximum(df["R_mean"], df["B_mean"])
    )

    df["normalized_green_red_difference"] = (
        (df["G_mean"] - df["R_mean"]) /
        (df["G_mean"] + df["R_mean"] + eps)
    )

    df["normalized_green_blue_difference"] = (
        (df["G_mean"] - df["B_mean"]) /
        (df["G_mean"] + df["B_mean"] + eps)
    )

    # =====================================================
    # Texture stress features
    # =====================================================

    df["texture_stress_index"] = (
        df["gray_std"] +
        df["entropy"] +
        df["glcm_contrast"]
    ) / 3

    df["texture_uniformity_index"] = (
        df["glcm_energy"] +
        df["glcm_homogeneity"]
    ) / 2

    df["roughness_to_uniformity_ratio"] = (
        df["texture_stress_index"] /
        (df["texture_uniformity_index"] + eps)
    )

    # =====================================================
    # Gray intensity features
    # =====================================================

    df["gray_contrast_ratio"] = (
        df["gray_std"] /
        (df["gray_mean"] + eps)
    )

    df["gray_brightness_normalized"] = (
        df["gray_mean"] / 255.0
    )

    # =====================================================
    # Clean invalid values
    # =====================================================

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def create_leaf_health_target(df):
    """
    Create image-based Leaf Health Index target.

    This replaces the older 'chlorophyll_proxy' wording.

    Important:
    This is NOT laboratory-measured chlorophyll.
    It is a visual health index based on visible greenness.

    Formula:
        normalized (G - R) / (G + R)

    Output:
        df with 'leaf_health_index_target'
    """

    df = df.copy()

    required_cols = ["G_mean", "R_mean"]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns for Leaf Health Index target: {missing_cols}"
        )

    eps = 1e-6

    df["leaf_health_index_target"] = (
        (df["G_mean"] - df["R_mean"]) /
        (df["G_mean"] + df["R_mean"] + eps)
    )

    min_val = df["leaf_health_index_target"].min()
    max_val = df["leaf_health_index_target"].max()

    df["leaf_health_index_target"] = (
        df["leaf_health_index_target"] - min_val
    ) / (
        max_val - min_val + eps
    )

    df["leaf_health_index_target"] = df["leaf_health_index_target"].clip(0, 1)

    return df


def prepare_features_for_training(df, add_target=False):
    """
    Main helper used before model training.

    Steps:
    1. Add engineered visual features.
    2. Optionally add Leaf Health Index target.

    Use:
        df = prepare_features_for_training(df, add_target=True)
    """

    df = add_visual_health_features(df)

    if add_target:
        df = create_leaf_health_target(df)

    return df
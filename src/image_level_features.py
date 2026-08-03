import pandas as pd
import numpy as np


def create_image_level_features(df):
    """
    Convert patch-level features into image-level features.

    Input:
        df = patch-level dataframe with image_id, label, x, y

    Output:
        image_df = one row per image
    """

    required_cols = ["image_id", "label", "x", "y"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Remove non-feature columns
    feature_cols = [
        col for col in df.columns
        if col not in ["image_id", "label", "x", "y"]
    ]

    numeric_df = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    df_clean = pd.concat(
        [
            df[["image_id", "label"]].reset_index(drop=True),
            numeric_df.reset_index(drop=True)
        ],
        axis=1
    )

    grouped = df_clean.groupby(["image_id", "label"])

    image_features = grouped.agg(["mean", "std", "min", "max"])

    # Flatten multi-index columns
    image_features.columns = [
        f"{feature}_{stat}"
        for feature, stat in image_features.columns
    ]

    image_features = image_features.reset_index()

    # Fill NaN from std if an image has only one valid patch
    image_features = image_features.fillna(0)

    return image_features
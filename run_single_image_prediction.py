import os
import json
import cv2
import joblib
import random
import numpy as np
import pandas as pd

from src.preprocessing import (
    segment_leaf,
    suppress_veins,
    normalize_leaf_color
)

from src.feature_extraction import extract_features_at_pixel
from src.data_loader import load_image
from src.recommendation import final_mangrove_recommendation
from src.config import MODEL_DIR, PATCH_SIZE
from src.feature_engineering import add_visual_health_features


# =========================================================
# Load patch-level species model
# =========================================================

species_model = joblib.load(
    os.path.join(MODEL_DIR, "species_model.pkl")
)

species_scaler = joblib.load(
    os.path.join(MODEL_DIR, "species_scaler.pkl")
)

species_label_encoder = joblib.load(
    os.path.join(MODEL_DIR, "species_label_encoder.pkl")
)


# =========================================================
# Load leaf health model
# =========================================================

health_model = joblib.load(
    os.path.join(MODEL_DIR, "leaf_health_model.pkl")
)

health_scaler = joblib.load(
    os.path.join(MODEL_DIR, "leaf_health_scaler.pkl")
)

health_feature_cols = joblib.load(
    os.path.join(MODEL_DIR, "leaf_health_feature_cols.pkl")
)


# =========================================================
# Optional image-level species model
# =========================================================

image_species_model_path = os.path.join(
    MODEL_DIR,
    "image_species_model.pkl"
)

image_species_scaler_path = os.path.join(
    MODEL_DIR,
    "image_species_scaler.pkl"
)

image_species_label_encoder_path = os.path.join(
    MODEL_DIR,
    "image_species_label_encoder.pkl"
)

image_species_feature_cols_path = os.path.join(
    MODEL_DIR,
    "image_species_feature_cols.pkl"
)

USE_IMAGE_LEVEL_SPECIES_MODEL = (
    os.path.exists(image_species_model_path)
    and os.path.exists(image_species_scaler_path)
    and os.path.exists(image_species_label_encoder_path)
    and os.path.exists(image_species_feature_cols_path)
)

if USE_IMAGE_LEVEL_SPECIES_MODEL:
    image_species_model = joblib.load(
        image_species_model_path
    )

    image_species_scaler = joblib.load(
        image_species_scaler_path
    )

    image_species_label_encoder = joblib.load(
        image_species_label_encoder_path
    )

    image_species_feature_cols = joblib.load(
        image_species_feature_cols_path
    )

else:
    image_species_model = None
    image_species_scaler = None
    image_species_label_encoder = None
    image_species_feature_cols = None


# =========================================================
# Helper functions
# =========================================================

def confidence_level(conf):
    if conf >= 0.75:
        return "High"
    elif conf >= 0.55:
        return "Medium"
    else:
        return "Low"


def explain_confidence(conf_level):
    if conf_level == "High":
        return (
            "High confidence means the model found strong species-level visual patterns."
        )
    elif conf_level == "Medium":
        return (
            "Medium confidence means the result is usable for preliminary screening, "
            "but field verification is still recommended."
        )
    else:
        return (
            "Low confidence means the image features were mixed or unclear. "
            "Use this result only as a preliminary suggestion."
        )


def explain_leaf_health_index(score):
    if score > 0.75:
        return "The leaf appears visually healthy with strong greenness indicators."
    elif score > 0.50:
        return "The leaf shows mild visual stress. Early monitoring is recommended."
    elif score > 0.30:
        return "The leaf shows moderate visual stress. Field inspection is recommended."
    else:
        return "The leaf shows severe visual stress. Immediate attention is recommended."


def image_quality_check(img_rgb):
    """
    Basic image quality check using blur and brightness.
    """

    gray = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2GRAY
    )

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    brightness = gray.mean()

    issues = []

    if blur_score < 15:
        issues.append("Image may be blurry.")

    if brightness < 50:
        issues.append("Image may be too dark.")

    if brightness > 220:
        issues.append("Image may be overexposed.")

    if len(issues) == 0:
        quality_label = "Good"
    else:
        quality_label = "Needs Review"

    return {
        "quality_label": quality_label,
        "blur_score": round(float(blur_score), 2),
        "brightness": round(float(brightness), 2),
        "issues": issues
    }


def create_sampled_region_visualization(
    img_rgb,
    sampled_points,
    output_path
):
    """
    Save image with sampled patch points marked.
    """

    vis_img = img_rgb.copy()

    for (x, y) in sampled_points:
        cv2.circle(
            vis_img,
            (x, y),
            2,
            (255, 0, 0),
            -1
        )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            vis_img,
            cv2.COLOR_RGB2BGR
        )
    )

    return output_path


def aggregate_single_image_features(feature_df):
    """
    Convert patch-level features from one uploaded image
    into one image-level feature row.

    This must match the aggregation used in image-level training:
    mean, std, min, max for every patch feature.
    """

    temp_df = feature_df.copy()

    temp_df["image_id"] = "uploaded_image"
    temp_df["label"] = "unknown"

    feature_cols = [
        col for col in temp_df.columns
        if col not in ["image_id", "label", "x", "y"]
    ]

    grouped = temp_df.groupby(
        ["image_id", "label"]
    )[feature_cols].agg(
        ["mean", "std", "min", "max"]
    )

    grouped.columns = [
        f"{feature}_{stat}"
        for feature, stat in grouped.columns
    ]

    image_row = grouped.reset_index()
    image_row = image_row.fillna(0)

    return image_row


def predict_species_from_image_level_model(feature_df):
    """
    Preferred species prediction method.
    Uses one aggregated row for the full image.
    """

    image_feature_row = aggregate_single_image_features(
        feature_df
    )

    X_image_species = image_feature_row.drop(
        columns=["image_id", "label"]
    )

    # Ensure same feature columns as training
    for col in image_species_feature_cols:
        if col not in X_image_species.columns:
            X_image_species[col] = 0

    X_image_species = X_image_species[
        image_species_feature_cols
    ]

    X_image_species = (
        X_image_species
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    X_image_species_scaled = image_species_scaler.transform(
        X_image_species
    )

    species_pred = image_species_model.predict(
        X_image_species_scaled
    )[0]

    species_probs = image_species_model.predict_proba(
        X_image_species_scaled
    )[0]

    final_species = image_species_label_encoder.inverse_transform(
        [species_pred]
    )[0]

    species_probability_dict = {
        image_species_label_encoder.classes_[i]: round(float(species_probs[i]), 3)
        for i in range(len(image_species_label_encoder.classes_))
    }

    species_confidence = float(
        np.max(species_probs)
    )

    return (
        final_species,
        species_confidence,
        species_probability_dict,
        "Image-Level Model"
    )


def predict_species_from_patch_level_model(feature_df):
    """
    Fallback species prediction method.
    Uses average probabilities from patch-level predictions.
    """

    X_species = feature_df.drop(
        columns=["x", "y"]
    )

    X_species = (
        X_species
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    X_species_scaled = species_scaler.transform(
        X_species
    )

    species_preds = species_model.predict(
        X_species_scaled
    )

    species_probs = species_model.predict_proba(
        X_species_scaled
    )

    species_labels = species_label_encoder.inverse_transform(
        species_preds
    )

    final_species = pd.Series(
        species_labels
    ).mode()[0]

    avg_species_probs = species_probs.mean(axis=0)

    species_probability_dict = {
        species_label_encoder.classes_[i]: round(float(avg_species_probs[i]), 3)
        for i in range(len(species_label_encoder.classes_))
    }

    species_confidence = float(
        np.max(avg_species_probs)
    )

    return (
        final_species,
        species_confidence,
        species_probability_dict,
        "Patch-Level Fallback Model"
    )


# =========================================================
# Main prediction function
# =========================================================

def predict_single_image(
    image_path,
    samples_per_image=300,
    random_state=42
):
    """
    Predict species, estimate Leaf Health Index,
    and generate recommendation for one uploaded mangrove leaf image.

    Uses image-level species model if available.
    Falls back to patch-level model otherwise.
    """

    # -----------------------------------------------------
    # Stable prediction seed
    # -----------------------------------------------------

    random.seed(random_state)
    np.random.seed(random_state)

    # -----------------------------------------------------
    # Validate image path
    # -----------------------------------------------------

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    img = load_image(image_path)

    if img is None:
        raise ValueError(
            "Failed to load image."
        )

    # -----------------------------------------------------
    # Image quality check
    # -----------------------------------------------------

    quality_report = image_quality_check(img)

    # -----------------------------------------------------
    # Segment leaf
    # -----------------------------------------------------

    mask = segment_leaf(img)

    if np.sum(mask > 0) == 0:
        raise ValueError(
            "No leaf region detected."
        )

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    leaf_rgb = cv2.bitwise_and(
        img,
        img,
        mask=mask
    )

    gray_leaf = cv2.cvtColor(
        leaf_rgb,
        cv2.COLOR_RGB2GRAY
    )

    gray_supp = suppress_veins(
        gray_leaf
    )

    leaf_norm = normalize_leaf_color(
        leaf_rgb
    )

    h, w = gray_supp.shape

    # -----------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------

    features_list = []
    sampled_points = []

    collected = 0
    attempts = 0
    max_attempts = samples_per_image * 10

    while (
        collected < samples_per_image
        and attempts < max_attempts
    ):
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
            f["x"] = x
            f["y"] = y

            features_list.append(f)
            sampled_points.append((x, y))

            collected += 1

    if len(features_list) == 0:
        raise ValueError(
            "No valid patches extracted."
        )

    feature_df = pd.DataFrame(
        features_list
    )
    feature_df = add_visual_health_features(feature_df)

    # =====================================================
    # Species prediction
    # =====================================================

    if USE_IMAGE_LEVEL_SPECIES_MODEL:
        (
            final_species,
            species_confidence,
            species_probability_dict,
            species_prediction_mode
        ) = predict_species_from_image_level_model(
            feature_df
        )

    else:
        (
            final_species,
            species_confidence,
            species_probability_dict,
            species_prediction_mode
        ) = predict_species_from_patch_level_model(
            feature_df
        )

    conf_level = confidence_level(
        species_confidence
    )

    # =====================================================
    # Leaf Health Index prediction
    # =====================================================

    X_health = feature_df[
        health_feature_cols
    ]

    X_health = (
        X_health
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    X_health_scaled = health_scaler.transform(
        X_health
    )

    health_preds = health_model.predict(
        X_health_scaled
    )

    leaf_health_index = float(
        np.mean(health_preds)
    )

    # =====================================================
    # Recommendation system
    # =====================================================

    recommendation = final_mangrove_recommendation(
        final_species,
        leaf_health_index
    )

    # =====================================================
    # Visualization
    # =====================================================

    os.makedirs(
        "outputs/predictions",
        exist_ok=True
    )

    vis_path = os.path.join(
        "outputs",
        "predictions",
        "sampled_regions_visualization.png"
    )

    create_sampled_region_visualization(
        img,
        sampled_points,
        vis_path
    )

    # =====================================================
    # Final result
    # =====================================================

    result = {
        "predicted_species": final_species,

        "species_prediction_mode": species_prediction_mode,

        "species_confidence": round(
            float(species_confidence),
            3
        ),

        "confidence_level": conf_level,

        "confidence_explanation": explain_confidence(
            conf_level
        ),

        "species_probabilities": species_probability_dict,

        "leaf_health_index": round(
            leaf_health_index,
            3
        ),

        "leaf_health_explanation": explain_leaf_health_index(
            leaf_health_index
        ),

        "health_status": recommendation["health_status"],

        "priority": recommendation["priority"],

        "species_guidance": recommendation["species_guidance"],

        "recommended_action": recommendation["recommended_action"],

        "visualization_image": vis_path,

        "image_quality": quality_report,

        "patches_used": len(features_list)
    }

    return result


# =========================================================
# CLI Entry
# =========================================================

if __name__ == "__main__":

    try:
        image_path = input(
            "Enter image path: "
        ).strip()

        result = predict_single_image(
            image_path
        )

        print(
            "\n========== Mangrove AI Prediction =========="
        )

        print(
            "Predicted Species:",
            result["predicted_species"]
        )

        print(
            "Species Prediction Mode:",
            result["species_prediction_mode"]
        )

        print(
            "Species Confidence:",
            result["species_confidence"]
        )

        print(
            "Confidence Level:",
            result["confidence_level"]
        )

        print(
            "Confidence Explanation:",
            result["confidence_explanation"]
        )

        print(
            "Leaf Health Index:",
            result["leaf_health_index"]
        )

        print(
            "Leaf Health Explanation:",
            result["leaf_health_explanation"]
        )

        print(
            "Health Status:",
            result["health_status"]
        )

        print(
            "Priority:",
            result["priority"]
        )

        print(
            "Patches Used:",
            result["patches_used"]
        )

        print("\nSpecies Probabilities:")

        for species, prob in result["species_probabilities"].items():
            print(f"{species}: {prob}")

        print("\nImage Quality:")

        print(result["image_quality"])

        print("\nSpecies Guidance:")

        print(
            result["species_guidance"]
        )

        print("\nRecommended Action:")

        print(
            result["recommended_action"]
        )

        print("\nVisualization Image:")

        print(
            result["visualization_image"]
        )

        # -------------------------------------------------
        # Save JSON result
        # -------------------------------------------------

        os.makedirs(
            "outputs/predictions",
            exist_ok=True
        )

        json_path = os.path.join(
            "outputs",
            "predictions",
            "single_image_result.json"
        )

        with open(json_path, "w") as f:
            json.dump(
                result,
                f,
                indent=4
            )

        print(
            f"\nResult saved to {json_path}"
        )

    except Exception as e:
        print("\nERROR:")
        print(str(e))
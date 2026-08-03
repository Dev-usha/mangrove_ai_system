import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import FIGURE_DIR, PREDICTION_DIR


def health_zone(score):
    if score > 0.75:
        return 3
    elif score > 0.50:
        return 2
    elif score > 0.30:
        return 1
    else:
        return 0


def health_zone_label(zone):
    if zone == 3:
        return "Healthy"
    elif zone == 2:
        return "Mild Stress"
    elif zone == 1:
        return "Moderate Stress"
    else:
        return "Severe Stress"


def create_health_maps(df, model, scaler, feature_cols):
    print("\n==============================")
    print("SPATIAL LEAF HEALTH MAPPING")
    print("==============================")

    X = df[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_scaled = scaler.transform(X)

    df = df.copy()

    df["predicted_leaf_health_index"] = model.predict(X_scaled)

    # Legacy column for old recommendation.py compatibility
    df["predicted_chlorophyll_proxy"] = df["predicted_leaf_health_index"]

    df["health_zone"] = df["predicted_leaf_health_index"].apply(health_zone)
    df["health_status"] = df["health_zone"].apply(health_zone_label)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["x"],
        df["y"],
        c=df["predicted_leaf_health_index"],
        cmap="YlGn",
        s=10,
        alpha=0.7
    )

    plt.colorbar(scatter, label="Predicted Leaf Health Index")
    plt.title("Spatial Leaf Health Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "spatial_leaf_health_map.png"),
        dpi=300
    )
    plt.show()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["x"],
        df["y"],
        c=df["health_zone"],
        cmap="RdYlGn",
        s=10,
        alpha=0.7
    )

    plt.colorbar(scatter, label="Health Zone: 0 Poor to 3 Healthy")
    plt.title("Mangrove Leaf Health Stress Zones")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "leaf_health_zone_map.png"),
        dpi=300
    )
    plt.show()

    prediction_path = os.path.join(
        PREDICTION_DIR,
        "mangrove_predictions.csv"
    )

    df.to_csv(prediction_path, index=False)

    print("Prediction CSV saved:", prediction_path)

    return df
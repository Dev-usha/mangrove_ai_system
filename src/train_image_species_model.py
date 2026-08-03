import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.config import MODEL_DIR, FIGURE_DIR, RANDOM_STATE
from src.image_level_features import create_image_level_features


def train_image_level_species_classifier(df):
    """
    Train species classifier using one row per image.
    This is more suitable for app-level prediction than patch-level classification.
    """

    print("\n==============================")
    print("IMAGE-LEVEL SPECIES MODEL")
    print("==============================")

    image_df = create_image_level_features(df)

    print("Image-level dataset shape:", image_df.shape)
    print("Images per class:")
    print(image_df["label"].value_counts())

    X = image_df.drop(columns=["image_id", "label"])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = image_df["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_val_scaled)

    acc = accuracy_score(y_val, y_pred)

    print("Image-Level Species Accuracy:", round(acc, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=label_encoder.classes_
        )
    )

    cm = confusion_matrix(y_val, y_pred)

    os.makedirs(FIGURE_DIR, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Greys")
    plt.title("Image-Level Species Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(label_encoder.classes_))

    plt.xticks(ticks, label_encoder.classes_, rotation=45)
    plt.yticks(ticks, label_encoder.classes_)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    fig_path = os.path.join(FIGURE_DIR, "image_level_species_confusion_matrix.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, "image_species_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "image_species_scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "image_species_label_encoder.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "image_species_feature_cols.pkl"))

    image_features_path = os.path.join("data", "processed", "image_level_features.csv")
    image_df.to_csv(image_features_path, index=False)

    print("Image-level species model saved.")
    print("Image-level features saved to:", image_features_path)

    return model, scaler, label_encoder, list(X.columns), image_df
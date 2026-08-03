import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import MODEL_DIR, FIGURE_DIR, RANDOM_STATE


def train_species_classifier(df):
    print("\n==============================")
    print("SPECIES IDENTIFICATION MODEL")
    print("==============================")

    X = df.drop(columns=["label", "x", "y", "image_id","leaf_health_index_target"],errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = df["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    groups = df["image_id"]

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_idx, val_idx = next(gss.split(X, y_encoded, groups=groups))

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y_encoded[train_idx]
    y_val = y_encoded[val_idx]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
)


    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_val_scaled)

    acc = accuracy_score(y_val, y_pred)

    print("Species Accuracy:", round(acc, 4))

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
    plt.title("Species Classification Confusion Matrix")
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

    fig_path = os.path.join(FIGURE_DIR, "species_confusion_matrix.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, "species_model.pkl"),compress=3)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "species_scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "species_label_encoder.pkl"))

    print("Species model saved.")

    return model, scaler, label_encoder
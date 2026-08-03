import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.config import MODEL_DIR, FIGURE_DIR, RANDOM_STATE
from src.feature_engineering import create_leaf_health_target


def train_leaf_health_model(df):
    """
    Train Random Forest model to predict image-based Leaf Health Index.

    Important:
    Leaf Health Index is not lab-measured chlorophyll.
    It is a visual health score based on greenness and stress-related image features.
    """

    print("\n==============================")
    print("LEAF HEALTH INDEX MODEL")
    print("==============================")

    df = df.copy()

    # If target does not exist, create it
    if "leaf_health_index_target" not in df.columns:
        df = create_leaf_health_target(df)

    exclude_cols = [
        "x",
        "y",
        "label",
        "image_id",
        "leaf_health_index_target",

        "R_mean",
        "G_mean",

        "green_ratio",
        "red_ratio",
        "excess_green",
        "red_green_diff",
        "green_red_ratio",
        "red_green_ratio",
        "greenness_strength",
        "normalized_green_red_difference",
        "normalized_green_blue_difference"
]

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
    ]

    X = df[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = df["leaf_health_index_target"]

    groups = df["image_id"]

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        gss.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", round(mse, 6))
    print("RMSE:", round(rmse, 6))
    print("R2 Score:", round(r2, 4))

    os.makedirs(FIGURE_DIR, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Actual Leaf Health Index")
    plt.ylabel("Predicted Leaf Health Index")
    plt.title("Leaf Health Index Prediction")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "leaf_health_prediction.png"),
        dpi=300
    )
    plt.show()

    residuals = y_test - y_pred

    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=40)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Leaf Health Index Residual Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "leaf_health_residuals.png"),
        dpi=300
    )
    plt.show()

    print("Residual Mean:", round(residuals.mean(), 6))
    print("Residual Std:", round(residuals.std(), 6))

    os.makedirs(MODEL_DIR, exist_ok=True)

    # New correct names
    joblib.dump(model, os.path.join(MODEL_DIR, "leaf_health_model.pkl"),compress=3)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "leaf_health_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "leaf_health_feature_cols.pkl"))

    # Legacy backup names so older code does not break
    #joblib.dump(model, os.path.join(MODEL_DIR, "chlorophyll_model.pkl"))
    #joblib.dump(scaler, os.path.join(MODEL_DIR, "chlorophyll_scaler.pkl"))
    #joblib.dump(feature_cols, os.path.join(MODEL_DIR, "chlorophyll_feature_cols.pkl"))

    print("Leaf health model saved.")

    return model, scaler, feature_cols, df


# Backward-compatible alias
def train_chlorophyll_model(df):
    return train_leaf_health_model(df)
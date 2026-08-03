import pandas as pd

from src.config import FEATURE_CSV
from src.feature_engineering import prepare_features_for_training
from src.train_species_model import train_species_classifier
from src.train_health_model import train_leaf_health_model
from src.health_mapping import create_health_maps
from src.recommendation import generate_recommendation_report

# ============================================================
# Load feature dataset
# ============================================================

df = pd.read_csv(FEATURE_CSV)

print("Loaded original feature dataset:", df.shape)

# ============================================================
# Feature engineering
# ============================================================

df = prepare_features_for_training(
    df,
    add_target=True
)

print("Feature-engineered dataset:", df.shape)

# ============================================================
# Train species model
# ============================================================

species_model, species_scaler, label_encoder = train_species_classifier(df)

# ============================================================
# Train Leaf Health Index model
# ============================================================

health_model, health_scaler, feature_cols, df_with_health = train_leaf_health_model(df)

# ============================================================
# Create health maps
# ============================================================

df_predictions = create_health_maps(
    df_with_health,
    health_model,
    health_scaler,
    feature_cols
)

# ============================================================
# Generate recommendation report
# ============================================================

report = generate_recommendation_report(df_predictions)

print("\nTraining and recommendation pipeline completed.")
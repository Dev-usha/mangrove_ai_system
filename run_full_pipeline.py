from src.data_loader import check_dataset_exists
from src.feature_extraction import create_feature_dataset
from src.train_species_model import train_species_classifier
from src.train_health_model import train_chlorophyll_model
from src.health_mapping import create_health_maps
from src.recommendation import generate_recommendation_report

print("Starting full Mangrove AI pipeline...")

check_dataset_exists()

df = create_feature_dataset()

print("Feature dataset ready:", df.shape)

species_model, species_scaler, label_encoder = train_species_classifier(df)

health_model, health_scaler, feature_cols, df_with_chl = train_chlorophyll_model(df)

df_predictions = create_health_maps(
    df_with_chl,
    health_model,
    health_scaler,
    feature_cols
)

recommendation_report = generate_recommendation_report(df_predictions)

print("\nFull pipeline completed successfully.")
print(recommendation_report)
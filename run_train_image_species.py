import pandas as pd

from src.config import FEATURE_CSV
from src.train_image_species_model import train_image_level_species_classifier

df = pd.read_csv(FEATURE_CSV)

print("Loaded patch feature dataset:", df.shape)

model, scaler, label_encoder, feature_cols, image_df = train_image_level_species_classifier(df)

print("\nImage-level species training completed.")

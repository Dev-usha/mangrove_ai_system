from src.data_loader import check_dataset_exists
from src.feature_extraction import create_feature_dataset

check_dataset_exists()

df = create_feature_dataset()

print("\nFeature extraction finished.")
print("Shape:", df.shape)
print(df.head())

print("\nClass distribution:")
print(df["label"].value_counts())
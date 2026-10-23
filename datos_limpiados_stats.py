from src.data.loader import load_dataset
from src.data.cleaner import clean_dataset

df = load_dataset()

cleaned_df = clean_dataset(df)

print("Original shape:", df.shape)
print("Cleaned shape:", cleaned_df.shape)
print("Missing values:", cleaned_df.isnull().sum().sum())
print("Exact duplicate rows:", cleaned_df.duplicated().sum())
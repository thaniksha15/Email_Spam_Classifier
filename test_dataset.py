import pandas as pd

file_path = "dataset/SMSSpamCollection"

df = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset loaded successfully!")
print()
print(df.head())
print()
print("Dataset shape:", df.shape)
print()
print(df["label"].value_counts())
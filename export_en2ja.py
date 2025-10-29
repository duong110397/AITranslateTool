from datasets import load_dataset
import pandas as pd

# Load train data for en-ja
ds = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="train")  # chú ý phần config "en-ja"
# If config "en-ja" doesn't exist, you can use split train and filter translation keys.

# Convert to DataFrame and get en and ja columns
df = pd.DataFrame(ds["translation"])
df = df.rename(columns={"en": "en", "ja": "ja"})

# If you want to get sample 3.000 rows
df = df.sample(10000, random_state=42)

# Export CSV
df.to_csv("data/train.csv", index=False, encoding="utf-8")
print("✅ Saved data/train.csv with", len(df), "rows")

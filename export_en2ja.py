from datasets import load_dataset
import pandas as pd

# Load dữ liệu train cho cặp en-ja
ds = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="train")  # chú ý phần config "en-ja"
# Nếu config "en-ja" không tồn tại, có thể dùng split train và filter translation keys.

# Chuyển sang DataFrame và lấy cột en và ja
df = pd.DataFrame(ds["translation"])
df = df.rename(columns={"en": "en", "ja": "ja"})

# Nếu muốn lấy sample 3.000 dòng
df = df.sample(10000, random_state=42)

# Xuất CSV
df.to_csv("data/train.csv", index=False, encoding="utf-8")
print("✅ Saved data/train.csv with", len(df), "rows")

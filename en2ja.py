from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

print("🌐 AI Translator (English ⇄ Japanese)")
print("Type 'exit' to quit.\n")

while True:
    text = input("Enter text: ")
    if text.lower() == "exit":
        break
    prompt = f"translate English to Japanese: {text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    print("→", tokenizer.decode(outputs[0], skip_special_tokens=True))

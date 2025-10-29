from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import warnings

# Hide unnecessary warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Load model & tokenizer
model_name = "./finetuned_mt5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Choose device M1 (MPS)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
model.eval()  # help inference faster, turn off dropout

print(f"🌐 AI Translator (English → Japanese) [device: {device}]")
print("Type 'exit' to quit.\n")

# Function to translate
def translate_en_ja(text: str) -> str:
    prompt = f"translate English to Japanese: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():  # don't calculate gradient => faster
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,  # beam search: better than greedy
            early_stopping=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Loop interactively
while True:
    text = input("Enter English text: ").strip()
    if text.lower() == "exit":
        break
    print("→", translate_en_ja(text))

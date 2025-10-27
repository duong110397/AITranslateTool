from transformers import MT5ForConditionalGeneration, MT5Tokenizer

model = MT5ForConditionalGeneration.from_pretrained("./finetuned_mt5")
tokenizer = MT5Tokenizer.from_pretrained("./finetuned_mt5")

while True:
    text = input("Enter text: ")
    if not text:
        break
    input_text = "translate English to Japanese: " + text
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    print("→", tokenizer.decode(outputs[0], skip_special_tokens=True))

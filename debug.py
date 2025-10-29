from transformers import MT5ForConditionalGeneration, MT5Tokenizer

model_dir = "./finetuned_mt5"
tokenizer = MT5Tokenizer.from_pretrained(model_dir)
model = MT5ForConditionalGeneration.from_pretrained(model_dir)

while True:
    text = input("Enter English text: ")
    if not text.strip():
        break

    input_text = "translate English to Japanese: " + text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128)
    print("→", tokenizer.decode(outputs[0], skip_special_tokens=True))

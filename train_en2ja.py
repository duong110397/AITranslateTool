from datasets import load_dataset
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("csv", data_files={"train": "data/train.csv"})
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset["validation"] = dataset.pop("test")

# Preprocess
def preprocess(example):
    prefix = "translate English to Japanese: "
    inputs = [prefix + ex for ex in example["en"]]
    targets = example["ja"]

    model_inputs = tokenizer(inputs, max_length=128, padding="longest", truncation=True)
    labels = tokenizer(targets, max_length=128, padding="longest", truncation=True)

    # Mask pad tokens
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in seq]
        for seq in labels["input_ids"]
    ]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=15,  # tăng vì dataset nhỏ
    save_total_limit=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./finetuned_mt5")
tokenizer.save_pretrained("./finetuned_mt5")

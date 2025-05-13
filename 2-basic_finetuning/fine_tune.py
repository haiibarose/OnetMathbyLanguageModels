# fine_tune.py
import torch
from torch.utils.data import DataLoader
from transformers import (
    XGLMTokenizer,
    XGLMForCausalLM,
    default_data_collator
)
from torch.optim import AdamW
from datasets import Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import datetime
import re
import json
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
# model_name = "facebook/xglm-2.9B"
model_name = "facebook/xglm-564M"
tokenizer = XGLMTokenizer.from_pretrained(model_name)
model = XGLMForCausalLM.from_pretrained(model_name).to(device)

# Full dataset
input_train_path = 'cleaned_onet_train.csv'
input_test_path = 'onet_math_test_filtered.jsonl'

df = pd.read_csv(input_train_path)
df = df.rename(columns={
    'choice 1': 'A.',
    'choice 2': 'B.',
    'choice 3': 'C.',
    'choice 4': 'D.',
    'choice 5': 'E.'
})

answer_map = {
    1: 'A.',
    2: 'B.',
    3: 'C.',
    4: 'D.',
    5: 'E.'
}
df['correct_answer'] = df['choice_ans'].map(answer_map)
choice_keys = ['A.', 'B.', 'C.', 'D.', 'E.']

raw_data = []
for _, row in df.iterrows():
    # Extract the question and choices
    question_text = str(row['question']).strip()

    choices = []
    for key in choice_keys:
        if pd.notna(row.get(key)):
            choices.append(f"{key} {row[key]}")

    correct_answer = str(row.get('correct_answer', '')).strip()

    # Append to the final list
    raw_data.append({
        "question": question_text,
        "choices": choices,
        "correct_answer": correct_answer
    })

raw_data_test = []
with open(input_test_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue  # Skip malformed lines

        # Extract question text
        question_text = data.get('question', '').strip()

        # Build choices list
        choices = []
        for key in choice_keys:
            if key in data:
                choices.append(f"{key} {data[key]}")

        # Extract the correct answer
        correct_answer = data.get('answer', '').strip()

        # Append to the result list
        raw_data_test.append({
            "question": question_text,
            "choices": choices,
            "correct_answer": correct_answer
        })

# Format input-target pairs
def format_example(example):
    input_text = example["question"] + "\n" + "\n".join(example["choices"]) + "\nคำตอบคือ "
    label_text = example["correct_answer"]
    full_text = input_text + label_text
    return {"text": full_text}

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Datasets
train_dataset = Dataset.from_list([format_example(x) for x in raw_data])
test_dataset = Dataset.from_list([format_example(x) for x in raw_data_test])
train_dataset = train_dataset.map(tokenize, remove_columns=["text"])
test_dataset = test_dataset.map(tokenize, remove_columns=["text"])

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=default_data_collator
)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training
model.train()
for epoch in range(200):
    print(f"Epoch {epoch+1}")
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Helper: extract answer letter from model output
def extract_choice_from_output(output_text, valid_choices):
    # Look for the part after "คำตอบคือ" or "คําตอบคือ"
    answer_start = re.search(r"(?:คำตอบคือ|คําตอบคือ)", output_text)
    if answer_start:
        answer_text = output_text[answer_start.end():]
    else:
        answer_text = output_text  # fallback

    # Find all valid choices and return the first one
    pattern = r'\b(' + '|'.join(re.escape(choice) for choice in valid_choices) + r')'
    matches = re.findall(pattern, answer_text)
    return matches[0] if matches else "N/A"

# Evaluation
def evaluate(model, tokenizer, data):
    model.eval()
    raw_outputs = []
    extracted_answers = []
    true_answers = []

    for example in data:
        input_text = example["question"] + "\n" + "\n".join(example["choices"]) + "\nคำตอบคือ "
        true_answer = example["correct_answer"]
        valid_choices = [c.split(".")[0] + "." for c in example["choices"]]  # ['A.', 'B.', ...]

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).to(device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        extracted = extract_choice_from_output(decoded_output, valid_choices)

        raw_outputs.append(decoded_output.strip())
        extracted_answers.append(extracted)
        true_answers.append(true_answer)

    correct = sum([p == t for p, t in zip(extracted_answers, true_answers)])
    acc = correct / len(true_answers)
    return acc, raw_outputs, extracted_answers, true_answers

# Evaluate both models
original_model = XGLMForCausalLM.from_pretrained(model_name).to(device)
original_acc, original_raw, original_preds, original_labels = evaluate(original_model, tokenizer, raw_data_test)
fine_tuned_acc, fine_tuned_raw, fine_tuned_preds, fine_tuned_labels = evaluate(model, tokenizer, raw_data_test)

# Save results
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"results_{timestamp}_564M_200epochs_new_train.txt"

with open(filename, "w", encoding="utf-8") as f:
    f.write("=== Evaluating Original Pretrained Model ===\n")
    for i, (raw, pred, true) in enumerate(zip(original_raw, original_preds, original_labels)):
        f.write(f"Question {i+1}:\n")
        f.write(f"Raw Output: {raw}\n")
        f.write(f"Extracted Answer: {pred}\n")
        f.write(f"True Answer: {true}\n\n")
    f.write(f"Original Model Accuracy: {original_acc:.2f}\n\n")

    f.write("=== Evaluating Fine-tuned Model ===\n")
    for i, (raw, pred, true) in enumerate(zip(fine_tuned_raw, fine_tuned_preds, fine_tuned_labels)):
        f.write(f"Question {i+1}:\n")
        f.write(f"Raw Output: {raw}\n")
        f.write(f"Extracted Answer: {pred}\n")
        f.write(f"True Answer: {true}\n\n")
    f.write(f"Fine-tuned Model Accuracy: {fine_tuned_acc:.2f}\n\n")

    f.write("=== Performance Comparison ===\n")
    f.write(f"Original Pretrained Accuracy:      {original_acc:.2f}\n")
    f.write(f"Fine-tuned Model Accuracy:         {fine_tuned_acc:.2f}\n")
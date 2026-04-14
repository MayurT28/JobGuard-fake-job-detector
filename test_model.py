from transformers import BertForSequenceClassification, BertTokenizer
import torch

print("Loading model from local files...")
model = BertForSequenceClassification.from_pretrained('model/bert_fakejob')
tokenizer = BertTokenizer.from_pretrained('model/bert_fakejob')
model.eval()
print("Model loaded successfully!")

# Quick test
test_text = "Urgent hiring! Work from home. Earn 40000 per month. No experience needed. Pay registration fee of 500 to apply."

inputs = tokenizer(
    test_text,
    return_tensors='pt',
    truncation=True,
    max_length=256
)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    prediction = outputs.logits.argmax().item()

label = "FAKE" if prediction == 1 else "REAL"
confidence = probs[prediction].item()

print(f"\nTest posting verdict: {label}")
print(f"Confidence: {confidence*100:.1f}%")

# ─────────────────────────────────────────
# FULL DATASET EVALUATION (Accuracy + F1-score)
# ─────────────────────────────────────────

print("\nRunning dataset evaluation...")

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/cleaned_jobs.csv").sample(1800, random_state=42)

texts = df["combined_text"].tolist()
labels = df["fraudulent"].tolist()


class JobDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item


dataset = JobDataset(texts, labels)
loader = DataLoader(dataset, batch_size=16)

predictions = []
true_labels = []

with torch.no_grad():
    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        preds = outputs.logits.argmax(dim=1)

        predictions.extend(preds.tolist())
        true_labels.extend(batch["labels"].tolist())


accuracy = accuracy_score(true_labels, predictions)

print("\nAccuracy:", round(accuracy, 4))
print("\nClassification Report:\n")
print(classification_report(true_labels, predictions))
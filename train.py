import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import mlflow

# ─────────────────────────────────────────
# STEP 1 - Load cleaned data
# ─────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('data/cleaned_jobs.csv')

texts = df['combined_text'].tolist()
labels = df['fraudulent'].tolist()

# ─────────────────────────────────────────
# STEP 2 - Split into train / val / test
# ─────────────────────────────────────────
# First split off 10% for test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    texts, labels,
    test_size=0.10,
    random_state=42,          # random_state=42 means same split every run
    stratify=labels           # keep same fake/real ratio in each split
)

# Then split remaining into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.11,           # 0.11 of 90% ≈ 10% of total
    random_state=42,
    stratify=y_train_val
)

print(f"Train size: {len(X_train)}")
print(f"Val size:   {len(X_val)}")
print(f"Test size:  {len(X_test)}")

# ─────────────────────────────────────────
# STEP 3 - Tokenizer
# ─────────────────────────────────────────
print("\nLoading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ─────────────────────────────────────────
# STEP 4 - Dataset class
# ─────────────────────────────────────────
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
        item['labels'] = torch.tensor(self.labels[idx])
        return item

print("Tokenizing data (this takes 2-3 minutes)...")
train_dataset = JobDataset(X_train, y_train)
val_dataset   = JobDataset(X_val, y_val)
test_dataset  = JobDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16)
test_loader  = DataLoader(test_dataset,  batch_size=16)

print("Data ready.")
print("\nNext step: model loading — run this file fully to continue.")

# ─────────────────────────────────────────
# STEP 5 - Load BERT model
# ─────────────────────────────────────────
print("\nLoading BERT model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)
model.to(device)

# ─────────────────────────────────────────
# STEP 6 - Class weights (fixing imbalance)
# ─────────────────────────────────────────
# Remember the 4.8% fake problem?
# This tells the model "fake examples are rare,
# so when you see one — pay 20x more attention"
total = len(y_train)
fake_count = sum(y_train)
real_count = total - fake_count

weight_for_fake = real_count / fake_count   # ≈ 19.7
class_weights = torch.tensor([1.0, weight_for_fake]).to(device)

print(f"\nClass weight for fake: {weight_for_fake:.1f}x")

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# ─────────────────────────────────────────
# STEP 7 - Optimizer
# ─────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=2e-5)

# ─────────────────────────────────────────
# STEP 8 - Training loop
# ─────────────────────────────────────────
print("\nStarting training...")
print("Each epoch = model studies all 14,321 postings once")
print("We do 3 epochs = 3 full rounds of study\n")

EPOCHS = 3

for epoch in range(EPOCHS):
    # --- Training phase ---
    model.train()
    total_loss = 0

    for batch_num, batch in enumerate(train_loader):
        # Move batch to GPU/CPU
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch   = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = loss_fn(outputs.logits, labels_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print progress every 100 batches
        if batch_num % 100 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_num}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)

    # --- Validation phase ---
    model.eval()
    val_preds, val_true = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = outputs.logits.argmax(dim=1)
            val_preds.extend(preds.cpu().tolist())
            val_true.extend(batch['labels'].tolist())

    print(f"\nEpoch {epoch+1} complete")
    print(f"Average training loss: {avg_loss:.4f}")
    print(classification_report(
        val_true, val_preds,
        target_names=['Real', 'Fake']
    ))

# ─────────────────────────────────────────
# STEP 9 - Save the model
# ─────────────────────────────────────────
print("Saving model...")
model.save_pretrained('model/bert_fakejob')
tokenizer.save_pretrained('model/bert_fakejob')
print("Model saved to model/bert_fakejob/")

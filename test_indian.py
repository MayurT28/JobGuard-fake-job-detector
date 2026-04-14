import pandas as pd
from predict import predict

df = pd.read_csv('data/indian_fake_jobs.csv')

correct = 0
total = len(df)

for _, row in df.iterrows():
    label, confidence = predict(row['text'])
    actual = "FAKE" if row['fraudulent'] == 1 else "REAL"
    predicted = label
    if predicted == actual:
        correct += 1
    else:
        print(f"MISSED: {row['text'][:80]}...")
        print(f"  Predicted: {predicted} | Actual: {actual}\n")

accuracy = correct / total * 100
print(f"\nIndian dataset results:")
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {accuracy:.1f}%")
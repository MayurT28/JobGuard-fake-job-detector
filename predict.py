import torch
import requests
from transformers import BertForSequenceClassification, BertTokenizer

# ─────────────────────────────────────────
# 1. LOAD MODEL (once, when file is imported)
# ─────────────────────────────────────────
print("Loading model...")
model = BertForSequenceClassification.from_pretrained('model/bert_fakejob')
tokenizer = BertTokenizer.from_pretrained('model/bert_fakejob')
model.eval()
print("Model ready.")

# ─────────────────────────────────────────
# 2. BERT PREDICTION
# ─────────────────────────────────────────
def predict(text):
    inputs = tokenizer(
        text,
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
    return label, confidence

# ─────────────────────────────────────────
# 3. LLAMA EXPLANATION
# ─────────────────────────────────────────
def explain(text, label, confidence):
    prompt = f"""You are a job fraud detection assistant helping Indian job seekers.

A machine learning model analysed this job posting and marked it as {label} with {confidence:.0%} confidence.

Job posting:
{text[:600]}

In 4 - 5 sentences, explain specifically what patterns in this posting make it look {label.lower()}.
Mention exact suspicious phrases if fake. Be direct and simple — the reader is a regular Indian job seeker, not a tech expert.
Do not start with "I" or "The model"."""

    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()['response'].strip()

# ─────────────────────────────────────────
# 4. COMBINED FUNCTION (what app.py will call)
# ─────────────────────────────────────────
def analyse_job(text):
    label, confidence = predict(text)
    explanation = explain(text, label, confidence)
    return {
        "verdict": label,
        "confidence": f"{confidence*100:.1f}%",
        "explanation": explanation
    }

# ─────────────────────────────────────────
# 5. TEST IT
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        {
            "name": "Indian scam posting",
            "text": "Urgent hiring! Work from home data entry job. Earn 35000-40000 per month. No experience required. Training provided. Pay registration fee of 500 rupees to confirm your seat. Contact us on WhatsApp. XYZ Consultancy Pvt Ltd."
        },
        {
            "name": "Real job posting",
            "text": "Software Engineer at Infosys Pune. 3+ years Python experience required. B.Tech in Computer Science. Competitive salary as per industry standards. Apply through our official careers portal at infosys.com/careers."
        }
    ]

    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {case['name']}")
        print(f"{'='*50}")
        result = analyse_job(case['text'])
        print(f"Verdict:     {result['verdict']}")
        print(f"Confidence:  {result['confidence']}")
        print(f"Explanation: {result['explanation']}")

def combined_verdict(text):
    text_lower = text.lower()

    # Step 1 — BERT prediction
    label, confidence = predict(text)

    # Step 2 — classify signals
    strong_signals = []
    weak_signals = []

    # STRONG FRAUD SIGNALS
    if 'registration fee' in text_lower or 'security deposit' in text_lower:
        strong_signals.append("Mentions payment requirement")

    if 'whatsapp' in text_lower:
        strong_signals.append("Uses WhatsApp as primary contact method")

    if '@gmail.com' in text_lower or '@yahoo.com' in text_lower:
        strong_signals.append("Uses free email instead of company domain")

    suspicious_phrases = [
        'earn per day',
        'earn per week',
        'no interview required',
        'instant joining',
        'work from home data entry'
    ]
    if any(p in text_lower for p in suspicious_phrases):
        strong_signals.append("Unrealistic hiring claims")

    vague_hr = [
        'verification team',
        'onboarding team'
    ]
    if any(p in text_lower for p in vague_hr):
        strong_signals.append("Vague HR authority reference")

    # WEAK TRUST SIGNALS
    # WEAK TRUST SIGNALS

# Generic company description
    generic_company = [
        'our client',
        'a leading company',
        'a reputed firm',
        'we are hiring for it company'
    ]

    if any(p in text_lower for p in generic_company):
        weak_signals.append("Company name not clearly specified")


    # Only phone number provided
    import re
    phone_pattern = r'\b\d{10}\b'

    if re.search(phone_pattern, text_lower) and '@' not in text_lower:
        weak_signals.append("Only phone contact provided (no official email)")


    # Mass hiring language
    mass_hiring_phrases = [
        'immediate joiners required',
        'multiple openings',
        'bulk hiring'
    ]

    if any(p in text_lower for p in mass_hiring_phrases):
        weak_signals.append("Mass hiring language used")
    if any(p in text_lower for p in generic_company):
        weak_signals.append("Company identity not clearly specified")

    # WhatsApp without careers portal = stronger suspicion
    if "whatsapp" in text_lower and "careers" not in text_lower:
        strong_signals.append("No verifiable application channel")

    # Step 3 — adjust probability using ONLY strong signals
    strong_count = len(strong_signals)

    fake_probability = (
        confidence if label == "FAKE"
        else (1 - confidence)
    )

    adjusted_fake_prob = fake_probability + (strong_count * 0.08)
    adjusted_fake_prob = max(0.01, min(0.99, adjusted_fake_prob))

    # Step 4 — final verdict
    if adjusted_fake_prob >= 0.65:
        final_label = "FAKE"
        final_confidence = adjusted_fake_prob
    else:
        final_label = "REAL"
        final_confidence = 1 - adjusted_fake_prob

    return final_label, final_confidence, strong_signals, weak_signals
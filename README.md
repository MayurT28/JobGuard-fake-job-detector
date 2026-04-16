# JobGuard — Fake Job Posting Detector

JobGuard is an AI-powered system that detects potentially fraudulent job postings using a hybrid approach combining a fine-tuned BERT model with rule-based signal analysis and confidence calibration.

The system is designed for Indian job portal postings and provides real-time trust assessment through an interactive Streamlit interface.

---

## Live Demo

Access JobGuard here:
http://jobguard-demo.duckdns.org:8501

JobGuard detects potentially fake job postings using:
- BERT-based classification
- rule-based fraud signal detection
- optional LLM explanations (local mode with Ollama)

---

## Features

* Fake job detection using fine-tuned BERT transformer model
* Signal-based fraud indicator engine (WhatsApp contact, payment requests, urgency patterns, etc.)
* Confidence-aware risk classification (Green / Yellow / Red)
* Explanation layer powered by LLaMA for interpretable predictions
* Streamlit web interface for instant job posting evaluation
* Designed specifically for real-world job description inputs from portals like Naukri and LinkedIn

---

## Tech Stack

Python
PyTorch
Transformers (HuggingFace BERT)
Streamlit
Rule-based NLP signal scoring
LLaMA explanation engine

---

## Dataset

Trained on 17,880 job postings with additional India-focused fraud-pattern tuning.

---

## How It Works

Pipeline:

1. User pastes job description text
2. BERT model predicts fraud probability
3. Signal engine detects suspicious textual indicators
4. Confidence calibration adjusts risk score
5. LLaMA generates explanation summary
6. UI displays final trust classification

---

## Risk Classification Logic

Green → Likely legitimate posting
Yellow → Needs verification
Red → High-risk posting

---

## Example Use Cases

Detect WhatsApp-only recruiter scams
Identify payment-request job fraud
Flag vague consultancy-style hiring patterns
Assist job seekers before submitting personal documents

---

## Project Structure

```
JobGuard-fake-job-detector/
│
├── app.py
├── predict.py
├── preprocess.py
├── train.py
│
├── model/
├── data/
│
├── requirements.txt
└── README.md
```

---

## Run Locally

Install dependencies:

```
pip install -r requirements.txt
```

Start application:

```
streamlit run app.py
```

---

## Future Improvements

Docker containerization
AWS deployment
REST API inference endpoint
Extended recruiter verification layer

---
## API Usage

Start API server:

python -m uvicorn api:app --reload

Swagger docs available at:

http://127.0.0.1:8000/docs

## Author

Mayur Tonge

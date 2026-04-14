from fastapi import FastAPI
from predict import combined_verdict

app = FastAPI(title="JobGuard Fake Job Detection API")


@app.get("/")
def home():
    return {"message": "JobGuard API is running"}


@app.post("/predict")
def predict_job(text: str):
    label, confidence, strong_signals, weak_signals = combined_verdict(text)

    return {
        "label": label,
        "confidence": confidence,
        "strong_signals": strong_signals,
        "weak_signals": weak_signals
    }
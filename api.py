import json, os
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = os.getenv("MODEL_PATH", "outputs/lora_r16/model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).eval()
id2label = {int(k): v for k, v in model.config.id2label.items()}

cal_file = os.path.join(MODEL_PATH, "calibration.json")
T = json.load(open(cal_file))["temperature"] if os.path.exists(cal_file) else 1.0

app = FastAPI(title="Financial Sentiment API")


class Headline(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: Headline):
    enc = tokenizer(body.text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits / T, dim=-1)[0]
    k = int(probs.argmax())
    return {
        "label": id2label[k],
        "confidence": round(float(probs[k]), 4),
        "scores": {id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }

# Financial Sentiment Intelligence

Classifies financial headlines as **positive**, **negative**, or **neutral**.
RoBERTa-base fine-tuned with LoRA on finance tweets, with calibrated confidence,
a FastAPI backend, and a Gradio web app.

## Results (test set, 2,388 headlines never used in training)

| Model | Accuracy | Macro F1 |
|---|---|---|
| FinBERT, no fine-tuning (baseline) | 72.5% | 0.668 |
| **RoBERTa + LoRA (this project)** | **89.7%** | **0.871** |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.781 | 0.885 | 0.830 |
| positive | 0.857 | 0.857 | 0.857 |
| neutral | 0.940 | 0.913 | 0.926 |

- A model that always answers "neutral" would score about 66%.
- Costly mistakes are rare: only 3 bearish headlines were called bullish, and 14 bullish were called bearish.
- Calibration (temperature scaling, T = 1.15) cut the Expected Calibration Error from 0.020 to 0.012.

![Evaluation](reports/lora_r16_eval.png)

## Data
[zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment):
9,543 labelled finance tweets (65% neutral, 20% positive, 15% negative). I split off 10% of the
training set for validation (8,588 train / 955 validation) and kept the official 2,388-row
validation split as an untouched test set.

## Method
- Base model: `FacebookAI/roberta-base`
- LoRA on the attention layers (r=16, alpha=32, dropout 0.1): 1.18M trainable parameters, 0.94% of the model
- 4 epochs, learning rate 2e-4, batch size 32, fp16. Training took 84 seconds on a free Colab T4.
- Best epoch chosen by validation macro F1
- Temperature scaling fitted on the validation set only

## Error analysis
Of the 245 test mistakes, 228 involve the neutral class. The most confident mistakes show:
- **Market slang and idioms.** "Santa Claus rally" (positive) is predicted neutral.
- **Sarcasm** hides negative sentiment.
- **Bad or good events in a flat, factual tone**, such as a data breach reported without emotional words.
- **Debatable labels**, where humans would disagree too.

## Run it
```bash
pip install -r requirements.txt
uvicorn api:app --port 8000     # API docs at http://localhost:8000/docs
python app.py                   # web app
```
The model weights are hosted on Hugging Face: YOUR_HUGGING_FACE_LINK_HERE.
Put them in `outputs/lora_r16/model` (or set the `MODEL_PATH` environment variable).

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "Bank reports record loan defaults, shares plunge"}'
```

## Limitations
- Trained on short English finance tweets. Longer articles and other languages may do worse.
- Single training run, no hyperparameter search.
- Sentiment is not a price prediction. This is not financial advice.

## Files
`financial_sentiment_training.ipynb` training and evaluation · `api.py` FastAPI backend · `app.py` Gradio app
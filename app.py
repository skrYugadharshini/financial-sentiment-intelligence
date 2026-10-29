import torch
import gradio as gr
from api import tokenizer, model, id2label, T


def analyze(text):
    if not text.strip():
        return {}
    enc = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits / T, dim=-1)[0]
    return {id2label[i]: float(p) for i, p in enumerate(probs)}


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=3, label="Financial headline"),
    outputs=gr.Label(num_top_classes=3, label="Sentiment"),
    title="Financial Sentiment Intelligence",
    description="RoBERTa fine-tuned with LoRA on financial news tweets. Type a headline.",
    examples=[
        ["Tesla beats earnings, stock jumps 8%"],
        ["Bank reports record loan defaults, shares plunge"],
        ["Company to hold annual shareholder meeting on Tuesday"],
        ["Santa Claus rally"],
    ],
)

if __name__ == "__main__":
    demo.launch()

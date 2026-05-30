import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentClassifier:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv(
            "INTENT_MODEL_REPO",
            "swarnaliM/distilbert-intent-chatbot"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()

        label = self.model.config.id2label[pred_id]

        return label, confidence
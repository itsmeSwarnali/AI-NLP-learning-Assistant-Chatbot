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

    def predict(self, text):
        inputs = self.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

        # DistilBERT does not use token_type_ids
        inputs.pop("token_type_ids", None)
    
        with torch.no_grad():
            outputs = self.model(**inputs)
    
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
    
        intent = self.id2label[predicted_class.item()]
        return intent, confidence.item()

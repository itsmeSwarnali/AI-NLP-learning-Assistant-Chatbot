import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentClassifier:
    def __init__(self):
        self.model_path = "models/intent_classifier"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Safe label mapping
        self.id2label = getattr(self.model.config, "id2label", None)

        # Fallback if model config does not contain proper labels
        if not self.id2label:
            self.id2label = {
                0: "concept_explanation",
                1: "code_help",
                2: "project_guidance",
                3: "career_cv",
                4: "general_chat",
                5: "unknown"
            }

        # Sometimes config keys are strings, convert them to int
        self.id2label = {int(k): v for k, v in self.id2label.items()}

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        # DistilBERT does not need token_type_ids
        inputs.pop("token_type_ids", None)

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        confidence_tensor, predicted_class_tensor = torch.max(probs, dim=1)

        predicted_class = int(predicted_class_tensor.item())
        confidence = float(confidence_tensor.item())

        intent = self.id2label.get(predicted_class, "concept_explanation")

        return intent, confidence

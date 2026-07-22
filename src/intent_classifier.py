import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentClassifier:
    def __init__(self):
        self.model_path = "models/intent_classifier"
        self.model_available = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.id2label = {
            0: "concept_explanation",
            1: "code_help",
            2: "project_guidance",
            3: "career_guidance",
            4: "general_chat",
        }

        if os.path.isdir(self.model_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

                self.model.to(self.device)
                self.model.eval()

                if hasattr(self.model.config, "id2label"):
                    self.id2label = {
                        int(k): v for k, v in self.model.config.id2label.items()
                    }

                self.model_available = True

            except Exception as e:
                print(f"Intent classifier could not be loaded: {e}")
                self.model_available = False
        else:
            print("Intent classifier model folder not found. Using fallback intent.")

    def predict(self, text):
        if not text:
            return "concept_explanation", 1.0

        # Fallback if model is missing on Streamlit Cloud
        if not self.model_available:
            return self.fallback_predict(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

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

    def fallback_predict(self, text):
        text = text.lower()

        if any(word in text for word in ["code", "python", "error", "bug", "debug", "traceback"]):
            return "code_help", 0.70

        if any(word in text for word in ["project", "build", "app", "github", "streamlit"]):
            return "project_guidance", 0.70

        if any(word in text for word in ["job", "cv", "resume", "career", "internship"]):
            return "career_guidance", 0.70

        return "concept_explanation", 0.60

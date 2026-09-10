"""Local inference using the saved best DistilBERT; no training or downloads."""

from pathlib import Path
from threading import Lock

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts/transformer/best_model"


class SentimentPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.labels = {int(key): value.lower() for key, value in self.model.config.id2label.items()}
        if set(self.labels) != {0, 1} or set(self.labels.values()) != {"negative", "positive"}:
            raise ValueError("Saved model must map its two output classes to negative and positive.")
        # Streamlit shares cached resources across sessions; serialize GPU inference.
        self._lock = Lock()

    def predict(self, review: str):
        if not isinstance(review, str) or not review.strip():
            raise ValueError("Please enter a movie review.")
        with self._lock, torch.no_grad():
            # Check whitespace above, but pass the exact original input to the tokenizer.
            inputs = self.tokenizer(
                review, truncation=True, max_length=256, return_tensors="pt"
            ).to(self.device)
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Model returned non-finite probabilities.")
            predicted_id = int(probabilities.argmax().item())
            by_label = {self.labels[index]: float(probabilities[index].item()) for index in self.labels}
        return {
            "prediction": self.labels[predicted_id].title(),
            "confidence": by_label[self.labels[predicted_id]],
            "positive_probability": by_label["positive"],
            "negative_probability": by_label["negative"],
        }
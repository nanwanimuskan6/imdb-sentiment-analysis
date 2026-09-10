"""Experimental CPU-only NLI aspect analysis and pretrained irony detection.

No training, keyword-based sentiment assignment, or changes to the primary DistilBERT output.
Model cards document label order:
https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768
https://huggingface.co/cardiffnlp/twitter-roberta-base-irony
https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/irony/mapping.txt
"""

import re
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CACHE = Path(__file__).resolve().parents[2] / ".cache/huggingface/hub"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
IRONY_MODEL = "cardiffnlp/twitter-roberta-base-irony"
ASPECTS = {
    "Acting": "acting", "Story/Plot": "story", "Direction": "direction",
    "Cinematography": "cinematography", "Music/Sound": "music or sound",
    "Characters": "characters", "Ending": "ending", "Pacing": "pacing",
}
# Noun anchors are an evidence safeguard, not sentiment rules. NLI must also
# support the aspect and its polarity. Implicit aspects may be missed by design.
ASPECT_ANCHORS = {
    "Acting": ("acting", "actor", "actors", "actress", "actresses", "performance", "performances", "cast"),
    "Story/Plot": ("story", "plot", "storyline", "narrative", "screenplay", "script"),
    "Direction": ("direction", "director", "directing", "directed"),
    "Cinematography": ("cinematography", "visuals", "camera", "lighting", "photography"),
    "Music/Sound": ("music", "sound", "soundtrack", "score", "audio"),
    "Characters": ("character", "characters"),
    "Ending": ("ending", "endings", "finale", "conclusion"),
    "Pacing": ("pacing", "pace", "tempo"),
}


def has_aspect_evidence(unit, name):
    words = set(re.findall(r"[a-z]+", unit.lower()))
    return bool(words.intersection(ASPECT_ANCHORS[name]))


# Conservative NLI evidence gates, fixed before examples; not calibrated or IMDb-validated.
EVIDENCE_THRESHOLD = 0.70
MAX_UNITS = 24


UNCERTAIN = "Uncertain / Insufficient evidence"


def evidence_label(positive, negative):
    if positive and negative:
        return "Mixed"
    if positive:
        return "Positive"
    if negative:
        return "Negative"
    return UNCERTAIN


def review_units(review):
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Please enter a movie review.")
    # Linguistic segmentation only: no words are assigned sentiment by rules.
    units = [part.strip() for part in re.split(r"(?<=[.!?;])\s+|,\s*|\b(?:but|and|although|however)\b",
                                              review, flags=re.IGNORECASE) if part.strip()]
    return units[:MAX_UNITS], len(units) > MAX_UNITS


class NuancedAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, cache_dir=CACHE)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, cache_dir=CACHE).cpu().eval()
        self.lock = Lock()
        self.entailment_id = self.model.config.label2id.get("entailment")
        if self.model.config.num_labels != 3 or self.entailment_id is None:
            raise ValueError("Unexpected NLI model output labels.")

    def entailment(self, pairs):
        scores = []
        with self.lock, torch.no_grad():
            for start in range(0, len(pairs), 8):
                batch = pairs[start:start + 8]
                inputs = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                        padding=True, truncation="only_first",
                                        max_length=256, return_tensors="pt")
                probabilities = torch.softmax(self.model(**inputs).logits, dim=-1)
                scores.extend(probabilities[:, self.entailment_id].tolist())
        return scores

    def analyze(self, review):
        units, limited = review_units(review)
        mentions = [(unit, f"This text discusses the {aspect}.")
                    for unit in units for aspect in ASPECTS.values()]
        mention_scores = self.entailment(mentions)
        candidates = [(unit, name, aspect, mention_scores[i * len(ASPECTS) + j])
                      for i, unit in enumerate(units)
                      for j, (name, aspect) in enumerate(ASPECTS.items())
                      if has_aspect_evidence(unit, name)
                      and mention_scores[i * len(ASPECTS) + j] >= EVIDENCE_THRESHOLD]
        sentiment_pairs = []
        for unit, name, aspect, score in candidates:
            sentiment_pairs.extend([(unit, f"The {aspect} was good."),
                                    (unit, f"The {aspect} was bad.")])
        polarity = self.entailment(sentiment_pairs)
        evidence = {}
        for index, (unit, name, aspect, mention) in enumerate(candidates):
            pos, neg = polarity[2 * index:2 * index + 2]
            if max(pos, neg) < EVIDENCE_THRESHOLD:
                continue
            sentiment = "Positive" if pos > neg else "Negative"
            evidence.setdefault(name, []).append({
                "sentiment": sentiment, "text": unit, "mention_score": mention,
                "entailment_score": max(pos, neg),
            })
        aspects = []
        for name, rows in evidence.items():
            sentiments = {row["sentiment"] for row in rows}
            aspects.append({"aspect": name,
                            "sentiment": evidence_label("Positive" in sentiments, "Negative" in sentiments),
                            "evidence": rows})
        # Aspect evidence takes priority. General NLI evidence is a separate fallback;
        # it never reads DistilBERT confidence or applies word-specific sentiment rules.
        sentiments = {row["sentiment"] for rows in evidence.values() for row in rows}
        general_evidence = None
        label = evidence_label("Positive" in sentiments, "Negative" in sentiments)
        if not sentiments:
            positive, negative, unsure = self.entailment([
                (review, "This text expresses a positive opinion."),
                (review, "This text expresses a negative opinion."),
                (review, "The reviewer is unsure how they feel."),
            ])
            # Require absolute support and separation, not a forced-choice argmax.
            score = max(positive, negative)
            supported = (score >= EVIDENCE_THRESHOLD
                         and score - min(positive, negative) >= 0.20
                         and score - unsure >= 0.20)
            general_evidence = {"positive": positive, "negative": negative,
                                "unsure": unsure, "supported": supported}
            if supported:
                label = "Positive" if positive > negative else "Negative"
        return {
            "sentiment": label,
            "evidence_source": "aspects" if sentiments else (
                "overall_nli" if label != UNCERTAIN else "insufficient"),
            "general_evidence": general_evidence,
            "aspects": aspects,
            "weak_evidence": label == UNCERTAIN, "units_limited": limited,
            "note": "Experimental zero-shot interpretation; evidence thresholds are uncalibrated. "
                    "Explicit aspect-name evidence required; implicit aspects may be missed. "
                    "Up to 24 clauses, 256 tokens per NLI pair. No IMDb aspect benchmark evaluation.",
        }


class IronyAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(IRONY_MODEL, cache_dir=CACHE)
        self.model = AutoModelForSequenceClassification.from_pretrained(IRONY_MODEL, cache_dir=CACHE).cpu().eval()
        self.lock = Lock()
        if self.model.config.num_labels != 2:
            raise ValueError("Unexpected irony model output labels.")

    def analyze(self, review):
        if not isinstance(review, str) or not review.strip():
            raise ValueError("Please enter a movie review.")
        # TweetEval mapping: 0=non_irony, 1=irony. Never flip the sentiment prediction.
        with self.lock, torch.no_grad():
            inputs = self.tokenizer(review, truncation=True, max_length=256, return_tensors="pt")
            probabilities = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
            predicted = int(probabilities.argmax())
        return {"flag": "Likely" if predicted == 1 else "Unlikely",
                "irony_probability": float(probabilities[1]),
                "note": "Experimental TweetEval irony model; not validated on movie reviews. First 256 tokens."}

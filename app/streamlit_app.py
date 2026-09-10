"""IMDb app: trained binary classifier plus clearly separated experimental analysis."""

import logging
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from imdb_sentiment.inference import SentimentPredictor
from imdb_sentiment.nuanced_analysis import NuancedAnalyzer, IronyAnalyzer, NLI_MODEL, IRONY_MODEL, UNCERTAIN

LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading saved DistilBERT...")
def load_predictor():
    return SentimentPredictor()


@st.cache_resource(show_spinner="Loading experimental aspect model on CPU...")
def load_nuanced():
    return NuancedAnalyzer()


@st.cache_resource(show_spinner="Loading experimental irony model on CPU...")
def load_irony():
    return IronyAnalyzer()


def display_analysis(review):
    try:
        with st.spinner("Running DistilBERT..."):
            binary = load_predictor().predict(review)
    except Exception:
        LOGGER.exception("Binary inference failed")
        st.error("Could not load or run the saved DistilBERT model. See the terminal for details.")
        return

    nuanced, irony = None, None
    try:
        with st.spinner("Inspecting aspect evidence on CPU..."):
            nuanced = load_nuanced().analyze(review)
    except Exception:
        LOGGER.exception("Nuanced analysis unavailable")
    try:
        with st.spinner("Checking experimental irony model on CPU..."):
            irony = load_irony().analyze(review)
    except Exception:
        LOGGER.exception("Irony analysis unavailable")

    st.subheader("Nuanced Analysis")
    if nuanced:
        uncertain = nuanced["sentiment"] == UNCERTAIN
        st.metric("Nuanced sentiment", "UNCERTAIN" if uncertain else nuanced["sentiment"].upper())
        if uncertain:
            st.info("Not enough reliable aspect-level or overall sentiment evidence was found for a nuanced interpretation.")
        elif nuanced["sentiment"] == "Mixed":
            st.info("Conflicting sentiment was detected across the identified aspects. "
                    "This may include opposing opinions about the same aspect.")
        st.caption("A separate experimental interpretation based on supported aspect or overall NLI evidence, "
                   "not DistilBERT confidence.")
        if nuanced.get("evidence_source") == "overall_nli":
            st.caption("Supported overall sentiment was found, but no aspect-level sentiment was identified.")
        if nuanced["units_limited"]:
            st.warning("Nuanced analysis is limited to the first 24 clauses.")
    else:
        st.info("Nuanced analysis is unavailable. The binary prediction remains available below.")

    st.subheader("Aspect Sentiment")
    if nuanced and nuanced["aspects"]:
        st.table([{"Aspect": item["aspect"], "Sentiment": item["sentiment"]}
                  for item in nuanced["aspects"]])
        with st.expander("View supporting review excerpts"):
            for item in nuanced["aspects"]:
                st.markdown(f"**{item['aspect']}**")
                for evidence in item["evidence"]:
                    st.write(f"{evidence['sentiment']}: {evidence['text']}")
    else:
        st.info("No aspects with sufficient model evidence were identified.")
    st.caption("Explicit aspect names and NLI evidence are required; implicit aspects can be missed.")

    st.subheader("DistilBERT Binary Prediction")
    st.markdown(f"**Prediction: {binary['prediction']}**")
    st.metric("Softmax confidence", f"{binary['confidence']:.2%}")
    for label in ("positive", "negative"):
        probability = binary[f"{label}_probability"]
        st.write(f"{label.title()} probability: {probability:.2%}")
        st.progress(probability)
    st.caption("Softmax probabilities represent model confidence and are not calibrated certainty.")
    st.caption("DistilBERT uses the first 256 tokens.")

    st.subheader("Sarcasm / Irony Check")
    st.write(f"**{'Not available' if irony is None else irony['flag']}**")
    if irony:
        st.write(f"Experimental irony probability: {irony['irony_probability']:.2%}")
        if irony["flag"] == "Likely":
            st.warning("Possible sarcasm/irony detected. The binary sentiment prediction may be unreliable.")
    st.caption("Experimental model trained on tweets. This flag never flips DistilBERT's prediction.")


def main():
    st.set_page_config(page_title="IMDb Movie Review Sentiment Analyzer", page_icon="🎬")
    st.title("IMDb Movie Review Sentiment Analyzer")
    st.write("Fine-tuned DistilBERT sentiment, with separate experimental aspect and irony analysis.")
    st.caption("The additional models run on CPU. Their first use downloads pretrained weights.")
    with st.form("review_form"):
        review = st.text_area("Your movie review", height=220,
                              placeholder="Type or paste a movie review here...")
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        if not review.strip():
            st.warning("Please enter a movie review before predicting.")
        else:
            display_analysis(review)
    st.subheader("Model Performance")
    with st.expander("Verified results and model details", expanded=True):
        st.markdown(
            "- **Primary model:** Fine-tuned DistilBERT\n"
            "- **Dataset:** IMDb 50K\n"
            "- **DistilBERT official test accuracy:** 90.652%\n"
            "- **DistilBERT test F1:** 0.909338\n"
            "- **LinearSVC test accuracy:** 90.312%"
        )
        st.markdown(f"- **Aspect/NLI model (CPU):** [{NLI_MODEL}](https://huggingface.co/{NLI_MODEL})\n"
                    f"- **Irony model (CPU):** [{IRONY_MODEL}](https://huggingface.co/{IRONY_MODEL})")
        st.caption("Existing benchmark metrics apply only to the primary classifier. Secondary "
                   "models have not been evaluated on IMDb aspect or sarcasm labels. NLI evidence "
                   "gates are fixed, uncalibrated heuristics over model scores, not keyword rules.")


if __name__ == "__main__":
    main()

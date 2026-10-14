from pathlib import Path
import joblib
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "news_classifier.joblib"

st.set_page_config(page_title="News Headline Classifier", page_icon="📰")
st.title("News Headline Classifier")
st.write("Enter an English news headline to predict its category.")

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model file not found. Include models/news_classifier.joblib.")
    st.stop()

artifact = load_artifact()
model = artifact["model"]
headline = st.text_area("News headline", height=130)

if st.button("Classify headline", type="primary"):
    if not headline.strip():
        st.warning("Enter a headline first.")
    else:
        prediction = model.predict([headline])[0]
        st.success(f"Predicted category: {prediction}")
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba([headline])[0]
            ranked = sorted(zip(model.classes_, scores), key=lambda x: x[1], reverse=True)[:5]
            st.dataframe(
                [{"Category": str(c), "Score": f"{s:.1%}"} for c, s in ranked],
                hide_index=True,
                use_container_width=True
            )

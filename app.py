import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('punkt',     quiet=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    tfidf    = joblib.load('models/tfidf_vectorizer.pkl')
    nb       = joblib.load('models/naive_bayes.pkl')
    svm      = joblib.load('models/svm.pkl')
    rf       = joblib.load('models/random_forest.pkl')
    ensemble = joblib.load('models/hybrid_ensemble.pkl')
    return tfidf, nb, svm, rf, ensemble

tfidf, nb_model, svm_model, rf_model, ensemble_model = load_models()

# ── Preprocessing ─────────────────────────────────────────────────────────────
stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Fake News Detection System")
st.markdown("""
    "**Hybrid AI Model** — Naive Bayes · SVM · Random Forest · Voting Ensemble  \n"
             """)
st.divider()

article = st.text_area(
    "Paste a news article below:",
    height=250,
    placeholder="Paste the full news article text here..."
)

st.divider()

if st.button("Analyse Article", type="primary", use_container_width=True):

    if not article.strip():
        st.warning("Please paste a news article before clicking Analyse.")

    elif len(article.split()) < 10:
        st.warning("Article is too short. Please paste a full news article.")

    else:
        # Preprocess and vectorize
        cleaned  = preprocess(article)
        features = tfidf.transform([cleaned])

        # ── Individual model predictions ──────────────────────────────────────
        nb_pred       = nb_model.predict(features)[0]
        svm_pred      = svm_model.predict(features)[0]
        rf_pred       = rf_model.predict(features)[0]
        ensemble_pred = ensemble_model.predict(features)[0]

        # Probabilities
        nb_proba       = nb_model.predict_proba(features)[0]
        rf_proba       = rf_model.predict_proba(features)[0]
        ensemble_proba = ensemble_model.predict_proba(features)[0]

        # ── Final verdict (ensemble) ──────────────────────────────────────────
        label      = "🔴 FAKE NEWS" if ensemble_pred == 0 else "🟢 REAL NEWS"
        confidence = ensemble_proba[ensemble_pred] * 100

        st.subheader("Final Verdict")
        if ensemble_pred == 0:
            st.error(f"**{label}**  —  Confidence: {confidence:.1f}%")
        else:
            st.success(f"**{label}**  —  Confidence: {confidence:.1f}%")

        st.divider()

        # ── Individual model breakdown ────────────────────────────────────────
        st.subheader("Individual Model Predictions")

        col1, col2, col3, col4 = st.columns(4)

        def prediction_label(pred):
            return "🔴 Fake" if pred == 0 else "🟢 Real"

        with col1:
            st.metric("Naive Bayes",
                      prediction_label(nb_pred),
                      f"{nb_proba[nb_pred]*100:.1f}%")

        with col2:
            st.metric("SVM",
                      prediction_label(svm_pred),
                      "LinearSVC")

        with col3:
            st.metric("Random Forest",
                      prediction_label(rf_pred),
                      f"{rf_proba[rf_pred]*100:.1f}%")

        with col4:
            st.metric("Hybrid Ensemble",
                      prediction_label(ensemble_pred),
                      f"{confidence:.1f}%")

        st.divider()

        # ── Probability breakdown ─────────────────────────────────────────────
        st.subheader("Ensemble Confidence Breakdown")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Probability — Real News", f"{ensemble_proba[1]*100:.1f}%")
        with col_b:
            st.metric("Probability — Fake News", f"{ensemble_proba[0]*100:.1f}%")

        st.progress(float(ensemble_proba[1]))

        st.divider()

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.caption(
            "⚠️ This system was trained on the ISOT Fake News Dataset and is intended "
            "for academic demonstration purposes. Results should not be used as the sole "
            "basis for determining whether a news article is fake or real."
        )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Project:** Hybrid AI Model for Fake News Detection  
    """)
    st.divider()
    st.header("Model Performance (ISOT)")
    st.markdown("""
    | Model | Accuracy |
    |---|---|
    | Naive Bayes | 94.95% |
    | SVM | 99.63% |
    | Random Forest | 99.63% |
    | **Hybrid Ensemble** | **99.40%** |
    """)
    st.divider()
    st.caption("Dataset: ISOT Fake News Dataset (Kaggle)")
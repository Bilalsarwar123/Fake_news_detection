import streamlit as st
import pickle
import re
import string

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("📰 Fake News Detection")
st.sidebar.info("""
Enter a news article below to check if it is **Real** or **Fake**.
- Model: Logistic Regression
- Text Vectorization: TF-IDF
""")

# Load model and vectorizer
try:
    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model: {e}")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# App title
st.title("📰 Fake News Detector")
st.markdown("Enter news content below and click **Check News** to detect fake news.")

# Text input area
news_input = st.text_area("📝 News Content", height=200)

# Button
if st.button("🔍 Check News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text")
    else:
        cleaned = clean_text(news_input)
        vect = vectorizer.transform([cleaned])

        # Prediction probabilities
        prob = model.predict_proba(vect)[0]
        pred = model.predict(vect)[0]

        if pred == 1:
            st.error(f"🚨 Fake News Detected\n**Confidence:** {prob[1]*100:.2f}%")
        else:
            st.success(f"✅ This News Appears to be Real\n**Confidence:** {prob[0]*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using **Python, Machine Learning & Streamlit**")

import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Custom CSS for modern UI
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    text-align: center;
    color: #1f2937;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #1e40af;
}
.result-box {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyze news and detect misinformation instantly</p>", unsafe_allow_html=True)

# Input card
st.markdown("### 📝 Enter News Content")
text = st.text_area("", height=150, placeholder="Paste news article here...")

# Button
if st.button("🔍 Analyze News"):
    if text.strip() != "":
        transformed = vectorizer.transform([text])
        prediction = model.predict(transformed)[0]

        if prediction == 0:
            st.markdown(
                "<div class='result-box' style='background-color:#fee2e2; color:#991b1b;'>🚫 Fake News Detected</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box' style='background-color:#dcfce7; color:#166534;'>✅ Real News</div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("Please enter some text")

# Footer
st.markdown("---")
st.caption("👨‍💻 Developed by Bilal Sarwar")
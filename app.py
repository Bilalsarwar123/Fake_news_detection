import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #f9fafb;
}
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #111827;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 30px;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}
.result {
    padding: 20px;
    border-radius: 12px;
    font-size: 20px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-title'>📰 Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze news content and detect misinformation using Machine Learning</div>", unsafe_allow_html=True)

# Layout columns
col1, col2 = st.columns([2,1])

# LEFT SIDE (Input)
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📝 Enter News Content")
    text = st.text_area("", height=200, placeholder="Paste your news article here...")

    if st.button("🔍 Analyze News"):
        if text.strip() != "":
            transformed = vectorizer.transform([text])
            prediction = model.predict(transformed)[0]

            if prediction == 0:
                st.markdown(
                    "<div class='result' style='background:#fee2e2;color:#991b1b;'>🚫 Fake News Detected</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='result' style='background:#dcfce7;color:#166534;'>✅ Real News</div>",
                    unsafe_allow_html=True
                )
        else:
            st.warning("Please enter some text")

    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT SIDE (Info Panel)
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("ℹ️ About Model")
    st.write("Model: Logistic Regression")
    st.write("Vectorizer: TF-IDF")
    st.write("Accuracy: ~95%")

    st.markdown("---")

    st.subheader("👨‍💻 Developer")
    st.write("Bilal Sarwar")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("🚀 Built with Streamlit | Fake News Detection Project")
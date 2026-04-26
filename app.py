import streamlit as st
import joblib
import streamlit as st

st.markdown("---")
st.markdown("👨‍💻 Developed by **Bilal Sarwar**")
import streamlit as st

st.image("ChatGPT Image Apr 26, 2026, 03_23_59 PM.png", width=150)

st.title("📰 Fake News Detection")
st.subheader("Developed by Bilal Sarwar")
# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.title("📰 Fake News Detection")

st.write("Enter news text below to check if it is Real or Fake.")

text = st.text_area("News Content")

if st.button("Check News"):
    if text.strip() != "":
        transformed = vectorizer.transform([text])
        prediction = model.predict(transformed)[0]

        if prediction == 0:
            st.error("🚫 Fake News")
        else:
            st.success("✅ Real News")
    else:
        st.warning("Please enter some text")
        
import streamlit as st
import joblib

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
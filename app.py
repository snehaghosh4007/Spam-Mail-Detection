
import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Mail Detection")
st.write("Enter an email or SMS message to check if it is Spam or Ham.")

message = st.text_area("Enter your message")

if st.button("Predict"):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        st.success("✅ Ham Mail")
    else:
        st.error("🚫 Spam Mail")

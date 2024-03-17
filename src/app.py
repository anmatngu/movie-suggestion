import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

loaded_model = joblib.load('src\\spam_detection_model.pkl')
feature_extraction = joblib.load('src\\feature_extraction.pkl')

def mail_detection(user_input):
    # Vectorize the email text
        input_mail = str(user_input)

        # convert text to feature vectors
        input_data_features = feature_extraction.transform([input_mail])

        # making prediction

        prediction = loaded_model.predict(input_data_features)
        spam_proba = loaded_model.predict_proba(input_data_features)[0][1]
        print(prediction)


        if prediction[0] == 1:
            st.write("This email is classified as SPAM.")
            st.write(f"Spam Probability: {(spam_proba):.2f}")
        else:
            st.write("This email is classified as Not SPAM.")
            st.write(f"Spam Probability: {(1 - spam_proba):.2f}")

def main():
    """
    Main function for the Streamlit app.
    """

    st.title("Spam Detection App")
    email_text = st.text_area("Enter Email Text:", height=200)

    if st.button("Classify"):
        mail_detection(email_text)

if __name__ == '__main__':
    main()
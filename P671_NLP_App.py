import streamlit as st
import pickle
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# Load Resources
# ----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

model = pickle.load(
    open('svc_sentiment_model.pkl','rb')
)

# LOAD TF-IDF VECTORIZER
vectorizer = pickle.load(
    open('tfidf_vectorizer.pkl','rb')
)


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ----------------------------
# Text Preprocessing Function
# ----------------------------
#def preprocess_text(text):
#    text = text.lower()
 #   text = re.sub(r"http\S+|www\S+", "", text)
#    text = re.sub(r"\d+", "", text)
 #   text = re.sub(r"[^a-zA-Z\s]", "", text)

 #   words = text.split()

 #   words = [
 #       lemmatizer.lemmatize(word)
#        for word in words
#        if word not in stop_words
 #   ]

 #   return " ".join(words)
    
def clean_text(text):

    # CONVERT TO LOWERCASE
    text = str(text).lower()

    # REMOVE URL
    text = re.sub(
        r'http\S+',
        '',
        text
    )

    # REMOVE HTML TAGS
    text = re.sub(
        r'<.*?>',
        '',
        text
    )

    # REMOVE SPECIAL CHARACTERS
    text = re.sub(
        r'[^a-zA-Z]',
        ' ',
        text
    )

    # TOKENIZATION
    words = text.split()

    # REMOVE SMALL WORDS
    words = [
        w for w in words
        if len(w) > 2
    ]

    # REMOVE STOPWORDS
    words = [
        w for w in words
        if w not in stop_words
    ]

    # LEMMATIZATION
    words = [lemmatizer.lemmatize(w)
             for w in words]

    # RETURN CLEAN TEXT
    return " ".join(words)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.info("""
### NLP Sentiment Analysis

**Model:** Support Vector Classifier (SVC)

**Vectorizer:** TF-IDF

**Tech Stack**
- Python
- Streamlit
- Scikit-Learn
- NLTK

Developed for sentiment classification of customer reviews.
""")

# ----------------------------
# Main Title
# ----------------------------
st.title("📊 Customer Review Sentiment Analysis")

st.markdown("""
Analyze customer reviews and classify them as **Positive** or **Negative** or **Neutral**
using a Machine Learning model.
""")

# SELECT MODE

mode = st.radio(

    "Choose Option",

    ["Input Text","Upload File"]

)
# ----------------------------
# Input Section
# ----------------------------

if mode == 'Input Text':

    # user review input
    text = st.text_area(
    "Enter Customer Review",
    height=180,
    placeholder="Type or paste a review here..."
    )  


    def clean_text(text):

        # CONVERT TO LOWERCASE
        text = str(text).lower()

        # REMOVE URL
        text = re.sub(
            r'http\S+',
            '',
            text
        )

        # REMOVE HTML TAGS
        text = re.sub(
            r'<.*?>',
            '',
            text
        )

        # REMOVE SPECIAL CHARACTERS
        text = re.sub(
            r'[^a-zA-Z]',
            ' ',
            text
        )

        # TOKENIZATION
        words = text.split()

        # REMOVE SMALL WORDS
        words = [
            w for w in words
            if len(w) > 2
        ]

        # REMOVE STOPWORDS
        words = [
            w for w in words
            if w not in stop_words
        ]

        # LEMMATIZATION
        words = [lemmatizer.lemmatize(w)
                 for w in words]

        # RETURN CLEAN TEXT
        return " ".join(words)
    
    # predict button
    if st.button("Predict"):

        # clean text
        clean = clean_text(text)

        # vectorization
        vec = vectorizer.transform(
            [clean]
        )

        # prediction
        prediction = model.predict(vec)[0]

        # ==================================
        # AUTO RATING BASED ON SENTIMENT
        # ==================================

        if prediction.lower() == 'positive':

            rating = 5
            emoji = "😊"

        elif prediction.lower() == 'negative':

            rating = 1
            emoji = "😞"

        else:

            rating = 3
            emoji = "😐"

        # ==================================
        # SHOW RESULT
        # ==================================

        st.success(
            f"Prediction : {prediction}"
        )

        # show auto rating
        st.info(
            f"Predicted Rating : {rating} ⭐"
        )

        # emoji
        st.write(
            f"{emoji} Sentiment Analysis Completed"
        )

# FILE UPLOAD MODE
# ==========================================

else:

    # FILE UPLOADER
    file = st.file_uploader(

        "Upload CSV or Excel File",

        type=['csv','xlsx']

    )

    # CHECK FILE
    if file is not None:

        # ==================================
        # READ FILE
        # ==================================

        if file.name.endswith('.csv'):

            df = pd.read_csv(file)

        else:

            df = pd.read_excel(file)

        # ==================================
        # SHOW ORIGINAL DATA
        # ==================================

        st.subheader(
            "Original Data"
        )

        st.write(
            df.head()
        )

        # ==================================
        # CREATE REVIEW COLUMN
        # ==================================

        if 'title' in df.columns and 'body' in df.columns:

            df['review'] = (

                df['title']
                .fillna('')
                .astype(str)

                + ' : ' +

                df['body']
                .fillna('')
                .astype(str)

            )

        elif 'review' in df.columns:

            df['review'] = df['review'].astype(str)

        else:

            st.error(

                "File must contain review column OR title/body columns"

            )

            st.stop()

        # ==================================
        # PROCESS FILE BUTTON
        # ==================================

        if st.button("Process File"):

            # REMOVE EMPTY REVIEWS
            df = df[
                df['review'].str.strip() != ""
            ]

            # REMOVE DUPLICATES
            df.drop_duplicates(

                subset='review',

                inplace=True

            )

            # REMOVE NON ENGLISH TEXT
            df = df[
                df['review'].str.contains(
                    r'[a-zA-Z]',
                    regex=True
                )
            ]

            # ==================================
            # TEXT CLEANING
            # ==================================

            df['clean_review'] = df['review'].apply(
                clean_text
            )

            # ==================================
            # REVIEW LENGTH
            # ==================================

            df['review_length'] = df['review'].apply(
                lambda x: len(x.split())
            )

            # ==================================
            # SHOW CLEANED DATA
            # ==================================

            st.subheader(
                "Cleaned Data"
            )

            st.write(

                df[
                    ['review','clean_review']
                ].head()

            )

            # ==================================
            # TF-IDF VECTORIZATION
            # ==================================

            vector = vectorizer.transform(

                df['clean_review']

            )

            # ==================================
            # PREDICTION
            # ==================================

            df['prediction'] = model.predict(
                vector
            )

            # ==================================
            # SHOW RESULT
            # ==================================

            st.subheader(
                "Prediction Result"
            )

            st.write(

                df[
                    ['review','prediction']
                ].head()

            )

            # ==================================
            # SENTIMENT COUNT
            # ==================================

            st.subheader(
                "Sentiment Count"
            )

            st.write(

                df['prediction'].value_counts()

            )

            # ==================================
            # BAR CHART
            # ==================================

            st.subheader(
                "Sentiment Distribution"
            )

            st.bar_chart(

                df['prediction'].value_counts()

            )

            # DOWNLOAD RESULT
            # ==================================

            csv = df.to_csv(

                index=False

            ).encode('utf-8')

            st.download_button(

                "Download Result",

                csv,

                "sentiment_prediction.csv",

                "text/csv"

            )


# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    """
    <center>
    <b>Machine Learning Sentiment Analysis Dashboard</b><br>
    Developed using Streamlit, TF-IDF, and SVC
    </center>
    """,
    unsafe_allow_html=True
)
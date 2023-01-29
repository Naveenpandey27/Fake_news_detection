import streamlit as st 
import joblib,os
import spacy
import pandas as pd

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

# Import matplotlib
import matplotlib
import matplotlib.pyplot as plt 

# Use matplotlib for rendering
matplotlib.use("Agg")

# Import Image and WordCloud libraries from PIL
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Load the vectorization object from disk
news_vectorizer = open("vectorization.pkl","rb")
news_cv = joblib.load(news_vectorizer)


def main():
    # Display the header
    st.markdown("<h1 style='text-align: center; color: White;'>Fake News Detection</h1>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='text-align: center;'>NLP and ML app with Streamlit</h2>
    """, unsafe_allow_html=True)

    # Sidebar activity selector
    activities = ['Prediction', 'NLP']
    choice = st.sidebar.selectbox('Choose Activity', activities)

    # Prediction section
    if choice == 'Prediction':
        st.info('Prediction with ML')
        # Text input field
        news_text = st.text_area('Enter Text')
        # Model selector
        all_ml_models = ['Logistic Regression', 'Decision Tree Classifier', 'Gradient boosting Classifier', 'Random Forest Classifier']
        model_choice = st.selectbox('Choose ML Model', all_ml_models)
        prediction_labels = {'Fake' : 0, 'Real': 1}
        # Predict button
        if st.button("Classify"):
            st.text("Original Text::\n{}".format(news_text))
            # Transform the input text
            vect_text = news_cv.transform([news_text]).toarray()
            # Load the selected model from disk
            if model_choice == 'Logistic Regression':
                predictor = joblib.load("LoR_model.pkl")
            elif model_choice == 'Decision Tree Classifier':
                predictor = joblib.load("Decision_TC_model.pkl")
            elif model_choice == 'Gradient boosting Classifier':
                predictor = joblib.load("GBC_model.pkl")
            elif model_choice == 'Random Forest Classifier':
                predictor = joblib.load("RFC_model.pkl")
                
            # Make the prediction
            prediction = predictor.predict(vect_text)
            prediction_label = prediction[0]
            if prediction_label == 0:
                st.success("Fake News")
            else:
                st.success("Real News")
                    
    # Check if the choice is NLP
    if choice == 'NLP':
        # Display a message to inform the user about the selected option
        st.info('Natural Language Processing')

        # Input for raw text
        raw_text = st.text_area('Enter Text')

        # List of NLP tasks to choose from
        nlp_task = ["Tokenization","Lemmatization","NER","POS Tags"]
        task_choice = st.selectbox("Choose NLP Task",nlp_task)
    
    # Check if the Analyze button is clicked
    if st.button("Analyze"):
        # Display the original text
        st.info("Original Text : \n{}".format(raw_text))

        # Process the text using spaCy NLP library
        doc = nlp(raw_text)
        
        # Check the chosen NLP task and perform the task
        if task_choice == 'Tokenization':
            result = [token.text for token in doc]
        elif task_choice == 'Lemmatization':
            result = ["'Token' : {}, 'Lemma' : {}".format(token.text,token.lemma_) for token in doc]
        elif task_choice == 'NER':
            result = [(entity.text, entity.label_) for entity in doc.ents]
        elif task_choice == 'POS Tags':
            result = ["'Token' : {}, 'POS' : {}, 'Dependency' : {}".format(word.text,word.tag_,word.dep_) for word in doc]
        
        # Display the result as a JSON
        st.json(result)

    # Check if the Tabulize button is clicked
    if st.button("Tabulize"):
        # Process the text using spaCy NLP library
        doc = nlp(raw_text)
        
        # Extract tokens, lemma, and POS information
        c_tokens = [token.text for token in doc]
        c_lemma = [token.lemma_ for token in doc]
        c_pos = [token.pos_ for token in doc]

        # Create a Pandas dataframe from the extracted information
        new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
        
        # Display the dataframe
        st.dataframe(new_df)

# Check if the script is being run as the main program
if __name__ == '__main__':
    main()

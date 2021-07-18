import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from collections import OrderedDict
from joblib import load


def data_preprocessing(input_json):
    
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import pickle
    from keras.preprocessing.text import Tokenizer
    import nltk
    Pkl_Filename = "tokenizer.pkl"
    with open(Pkl_Filename, 'rb') as file:
        Pickled_tokenizer_Model = pickle.load(file)
    nltk.download('stopwords')
    combined_text =  " The decription of the incident is {}. The incident happned in {} and {} area. The industry is {}. The potential level was {}. This was reported by a {}. The affected party was a {}. The year was {}. The month was {}. the day was {}".format(input_json["description"],input_json["country"],input_json["local"],input_json["industry_sector"],input_json["potential_accident_level"],input_json["gender"],input_json["employee_type"],input_json["year"],input_json["month"],input_json["day"])
    contractions = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
    }
    
    def clean_text(text, remove_stopwords=True):
        text = text.lower()
        if True:
            text = text.split()
            new_text = []
            for word in text:
                if word in contractions:
                    new_text.append(contractions[word])
                else:
                    new_text.append(word)
                text = " ".join(new_text)
                text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
                text = re.sub(r'\<a href', ' ', text)
                text = re.sub(r'&amp;', '', text)
                text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
                text = re.sub(r'<br />', ' ', text)
                text = re.sub(r'\'', ' ', text)
                if remove_stopwords:
                    text = text.split()
                    stops = set(stopwords.words("english"))
                    text = [w for w in text if not w in stops]
                    text = " ".join(text)
        return text    
    
    import re
    from nltk.corpus import stopwords 
    from pickle import dump, load
    cleaned_text = clean_text(combined_text)
    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 250
    # This is fixed.
    EMBEDDING_DIM = 100
    Pickled_tokenizer_Model.fit_on_texts(cleaned_text)
    word_index = Pickled_tokenizer_Model.word_index
    embedded_text = Pickled_tokenizer_Model.texts_to_sequences([cleaned_text])
    embedded_text = pad_sequences(embedded_text, maxlen=MAX_SEQUENCE_LENGTH)
    return embedded_text

def processinput():
    answers = []
    for key in st.session_state.keys():
            if(key.startswith('answer')):
                answers.append(st.session_state[key])

    #Convert this into the format needed for the model file!
    jon = {
    "country" : answers[0],
    "local" : answers[1],
    "description" : answers[7],
    "industry_sector" : answers[4],
    "potential_accident_level" : answers[5],
    "gender" : answers[2],
    "employee_type" : answers[3],
    "year" : "",
    "month" : "",
    "day" : ""}

    test_tt = data_preprocessing(jon)
    import tensorflow as tf
    new_model = tf.keras.models.load_model('capstone_dl_model')
    pred = new_model.predict(test_tt)
    labels = [1,2,3,4,5]
    return labels[np.argmax(pred)]


questions = ["**Which country you are from?**", "**Which Locality you are from?**", "**What is your gender?**","**What is your employment type?**","**Which Industrial sector?**",
            "**What is the potential accident level?**","**What is the Critical Risk?**","**Please describe the accident in detail**","**Based on the details you shared, it looks like the severity level is : **"]

size = len(questions)

if 'globalflag' not in st.session_state:
    st.session_state.globalflag = 0

if 'question' not in st.session_state:
    st.session_state.question = ''

if 'count' not in st.session_state:
    st.session_state.count = 0

counter = str(st.session_state.count)
previouscounter = str(st.session_state.count - 1)
if(st.session_state.count - 1 <= 0):
    previouscounter = str(0)    

html_temp = """
<div style="background-color:blue;padding:5px; border:1 px solid; border-radius:10; font-weight: bold; font-style:italic; ">
<h2 style="color:white;text-align:center; font-size:20px; font-bole:true;">Semi Ruled Chat Bot - Industrial accidents</h2>
</div></br>
"""
st.markdown(html_temp,unsafe_allow_html=True)


c1,c2 = st.beta_columns(2)

with c1:
    rollingtext = st.session_state.question
    placeholder = st.empty()    
        
    answer = st.text_input("Enter your Answer", value="", key=counter)

    with placeholder.beta_container():

        if(st.session_state.count < size):
            #Appending the answer to the chat text
            rollingtext = rollingtext+"\n\n"+st.session_state[previouscounter]

            #Store the answers to the session state
            answerkey = 'answers' + previouscounter
            st.session_state[answerkey] = st.session_state[previouscounter]

            #To add the next question to the rolling text
            nextquestion = questions[st.session_state.count]

            #Updating the state session with the rolling text
            rollingtext = rollingtext+"\r\n\n"+nextquestion

            #Predict the model if the details are all captured?
            if(st.session_state.count == 8):
                answers = processinput()
                rollingtext = rollingtext + "\r\n" + "**" + str(answers) + "**"

            st.session_state.question = rollingtext            
            st.markdown(rollingtext)

            #Incrementing the counter by 1
            st.session_state.count+=1

        else:
            st.markdown(rollingtext)

import streamlit as st
import pickle
import string
from sklearn import datasets
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
#copy from Jupyter Notebook file
def preprocessing(content):
    content = content.lower()
    content = nltk.word_tokenize(content)
    array = []
    for i in content:
        if i.isalnum():
            array.append(i)
    content = array[:]
    array.clear()
    for i in content:
        if i not in stopwords.words('english') and i not in string.punctuation:
            array.append(i)
    content = array[:]
    array.clear()
    for i in content:
        array.append(ps.stem(i))
    return " ".join(array)

#import model built in Jupyter Notebook
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

#user interface by Streamlit
st.title("Spam Classifier")
input = st.text_area("Your email content")

if st.button('Predict'):
    transformed = preprocessing(input)
    vector = tfidf.transform([transformed])
    result = model.predict(vector)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Ham")

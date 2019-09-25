from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import contractions
from bs4 import BeautifulSoup
import numpy as np
import re
import tqdm
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
from nltk import pos_tag
from nltk.stem import PorterStemmer


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        my_prediction = model.predict(tf_new.transform([data_preprocess_deploy(str(message))]))
    return render_template('result.html',prediction = my_prediction)

def data_preprocess_deploy(text):
    
    x = ''.join(" number " if c.isdigit() else c for c in text)
    x = x.replace('$' ,' currency ')
    x = x.replace('£' ,' currency ')
    x = x.replace('€' ,' currency ')
    x = x.replace('₹' ,' currency ')
    text = x.replace('¥' ,' currency ')
    
    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())

    tokens = [word for sent in nltk.sent_tokenize(text2) for word in
              nltk.word_tokenize(sent)]
    
    tokens = [word.lower() for word in tokens]
    
    stopwds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwds]
    
    tokens = [word for word in tokens if len(word)>=3]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    tagged_corpus = pos_tag(tokens)    
    
    Noun_tags = ['NN','NNP','NNPS','NNS']
    Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']

    lemmatizer = WordNetLemmatizer()
    
    def prat_lemmatize(token,tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token,'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token,'v')
        else:
            return lemmatizer.lemmatize(token,'n')
    
    pre_proc_text =  " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])             


    def strip_html_tags(text):
          soup = BeautifulSoup(text, "html.parser")
          [s.extract() for s in soup(['iframe', 'script'])]
          stripped_text = soup.get_text()
          stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
          return stripped_text

    def remove_accented_chars(text):
          text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
          return text
    
    doc = strip_html_tags(pre_proc_text)
    doc = doc.translate(doc.maketrans("\n\t\r", "   "))
    doc = doc.lower()
    doc = remove_accented_chars(doc)
    doc = contractions.fix(doc)
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = re.sub(' +', ' ', doc)
    doc = doc.strip()  
    return doc

if __name__ == '__main__':
    tf_new = joblib.load("tfidf.pkl")
    model = joblib.load("Logistic_spam_model.pkl")
    app.run(debug=True)
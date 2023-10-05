import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.replace('_', ' ')
    text = text.replace('|', ' ')  
    text = text.replace('[', ' ')  
    text = text.replace(']', ' ')  
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)
    return text

def preprocess(df):
    df['short_description_clean'] = df['short_description'].apply(clean_text)
    df['description_clean'] = df['description'].apply(clean_text)
    df['combined_clean'] = df['short_description_clean'] + ' ' + df['description_clean']
    tfidf_vectorizer = TfidfVectorizer(max_features=500000)  
    combined_tfidf = tfidf_vectorizer.fit_transform(df['combined_clean'])

    X_combined_train, X_combined_test, train_indices_combined, test_indices_combined = \
        train_test_split(combined_tfidf, df.index, test_size=0.2, random_state=42)

    return X_combined_train, X_combined_test, train_indices_combined, test_indices_combined


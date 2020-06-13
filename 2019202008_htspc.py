import numpy as np
import pandas as pd
from numpy.random import RandomState
from collections import Counter
from nltk.corpus import stopwords
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import string
from nltk.stem import PorterStemmer
import wordninja
import preprocessor as p
from nltk.stem import WordNetLemmatizer

if (__name__ == '__main__'):

    df = pd.read_csv('/content/data/1fe720be-90e4-4e06-9b52-9de93e0ea937_train.csv')
    tdf = pd.read_csv('/content/final_test_q1/fcac6286-6db1-4577-ad80-612fb9d36db9_test.csv')

    punct = string.punctuation
    trantab = str.maketrans(punct, len(punct)*' ')
    wordnet_lemmatizer = WordNetLemmatizer()
    stops = set(stopwords.words("english"))
    porter = PorterStemmer()

    df['text'] = df['text'].apply(lambda x: str(x).lower())
    p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.MENTION,p.OPT.NUMBER)
    df['text'] = df['text'].apply(lambda x: p.clean(str(x)).strip())
    df['text'] = df['text'].apply(lambda x: re.sub(r'#(\S+)', r' \1 ', str(x)))
    df['text'] = df['text'].str.strip(' "\'')
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)))
    df['text'] = df['text'].apply(lambda x : str(x).split())
    df['text'] = df['text'].apply(lambda x: [item for item in x if item not in stops and item is not ' '])
    df['text'] = df['text'].apply(lambda x: ( " ".join(x)))
    df['text'] = df['text'].apply(lambda x : re.sub(r"\b[a-zA-Z]\b", "", str(x)))
    df['text'] = df['text'].apply(lambda x : str(x).translate(trantab).strip())
    for i in range(df.shape[0]):
        temp = []
        for word in df['text'][i].split():
            w = wordnet_lemmatizer.lemmatize(word)
            temp.append(w)
        df['text'][i] = " ".join(temp)
    for i in range(df.shape[0]):
        text = df['text'][i]
        words = text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        df['text'][i]=" ".join(stemmed_words)


    tdf['text'] = tdf['text'].apply(lambda x: str(x).lower())
    p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.MENTION,p.OPT.NUMBER)
    tdf['text'] = tdf['text'].apply(lambda x: p.clean(str(x)).strip())
    tdf['text'] = tdf['text'].apply(lambda x: re.sub(r'#(\S+)', r' \1 ', str(x)))
    tdf['text'] = tdf['text'].str.strip(' "\'')
    tdf['text'] = tdf['text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)))
    tdf['text'] = tdf['text'].apply(lambda x : str(x).split())
    tdf['text'] = tdf['text'].apply(lambda x: [item for item in x if item not in stops and item is not ' '])
    tdf['text'] = tdf['text'].apply(lambda x: ( " ".join(x)))
    tdf['text'] = tdf['text'].apply(lambda x : re.sub(r"\b[a-zA-Z]\b", "", str(x)))
    tdf['text'] = tdf['text'].apply(lambda x : str(x).translate(trantab).strip())
    for i in range(tdf.shape[0]):
        temp = []
        for word in tdf['text'][i].split():
            w = wordnet_lemmatizer.lemmatize(word)
            temp.append(w)
        tdf['text'][i] = " ".join(temp)
    for i in range(tdf.shape[0]):
        text = tdf['text'][i]
        words = text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        tdf['text'][i]=" ".join(stemmed_words)


    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    x_t = tdf
    svm_rbf = Pipeline([('vect', TfidfVectorizer()),
                        ('clf', SVC(kernel='rbf', C=1.0)),
    ])
    rbf_model = svm_rbf.fit(x.text,y.values.ravel())
    predicted = rbf_model.predict(x_t.text)

    pred_df = pd.DataFrame({'labels':predicted})
    pred_df.to_csv('submission.csv',index=True)
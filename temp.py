import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

def init():
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('model/model.h5')
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    graph = tf.compat.v1.get_default_graph()
    return model, graph

global model, graph
model, graph = init()

df = pd.read_csv("Fake_News_Data.csv")
X = df['News']
y = df['Target']

voc_size = 10000

messages = X.copy()
nltk.download('stopwords')

"""Stemming"""

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages[i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

"""One Hot Representation"""

onehot_repr = [one_hot(words, voc_size) for words in corpus]

"""Embedding Representation"""

sent_len = 20
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_len)

import numpy as np

X_final = np.array(embedded_docs)
y_final = np.array(y)


from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X_final, y_final)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42, stratify=y_sm)



score = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
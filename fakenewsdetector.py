import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import nltk
import re
from nltk.corpus import stopwords


df = pd.read_csv("Fake_News_Data.csv")
X = df['News']
y = df['Target']

df_test = pd.read_csv("Fake_News_Data_Testing.csv")
A = df_test['News']
b = df_test['Target']

voc_size = 10000

messages = X.copy()
messages_test = A.copy()
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

corpus_test = []
for i in range(len(messages_test)):
    review = re.sub('[^a-zA-Z]', ' ', messages_test[i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus_test.append(review)

"""One Hot Representation"""

onehot_repr = [one_hot(words, voc_size) for words in corpus]
onehot_repr_test = [one_hot(words, voc_size) for words in corpus_test]

"""Embedding Representation"""

sent_len = 20
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_len)
embedded_docs_test = pad_sequences(onehot_repr_test, padding='pre', maxlen=sent_len)


import numpy as np
X_final = np.array(embedded_docs)
y_final = np.array(y)
A_final = np.array(embedded_docs_test)
b_final = np.array(b)


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
def generate_report(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


"""Model Building"""

from tensorflow.keras.layers import Dropout
embedding_vector_features=40
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_len))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


"""Data Balancing using SMOTE"""

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X_final, y_final)
A_sm, b_sm = smote.fit_resample(A_final, b_final)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42, stratify=y_sm)

"""Training Model"""

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=8, batch_size=64)

"""Performance Metrics and Accuracy"""

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""Prediction"""

y_pred = model.predict_classes(X_test)

"""Model Report Generation"""

print("Train Dataset Report")
generate_report(y_test, y_pred)

"""Testing"""

print("Test Dataset Report")
b_pred_test = model.predict_classes(A_sm)
generate_report(b_sm, b_pred_test)

from flask import Flask, request, render_template
from flask_cors import cross_origin
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from fakenewsdetector import model

"""Flask Framework Begins"""

ps = PorterStemmer()

app = Flask(__name__)


@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        news = (request.form["News"])
        corpus = []
        review = re.sub('[^a-zA-Z]', ' ', news)
        review = review.lower()
        review = review.split()
        print(review)

        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
        print(corpus)

        onehot_repr = [one_hot(words, 10000) for words in corpus]
        print(onehot_repr)
        embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=20)
        print(embedded_docs)
        prediction = model.predict(embedded_docs)
        print(prediction)

        if prediction > 0.1:
            output = "Real"
        else:
            output = "Fake"
        print(output)

        return render_template('index.html', prediction_text=f'This News Headline is {output}!')

    return render_template("index.html")


@app.route('/about')
@cross_origin()
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=False)

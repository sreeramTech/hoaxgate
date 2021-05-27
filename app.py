from flask import Flask, render_template, request, url_for, redirect
from numpy import load
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_bootstrap import Bootstrap
import pickle
import pandas as pd


app = Flask(__name__)
Bootstrap(app)
loaded_model = pickle.load(open('model.pkl', 'rb'))
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
dataframe = pd.read_csv("data/news.csv")
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)


def fake_news_detect(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    probability = loaded_model._predict_proba_lr(vectorized_input_data)
    probability = int(probability[0][0]*100)
    return prediction[0], probability


@app.route('/', methods=["POST", "GET"])
def index():
    if request.method == "POST":
        text = request.form['userTextInput']
        pred, prob = fake_news_detect(text)

        return render_template("result.html", res=pred, prob=prob)
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)

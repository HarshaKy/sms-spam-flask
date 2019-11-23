import pandas as pd
import numpy as  np
import pickle
# import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, url_for, request
# from sklearn.metrics import classification_report

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('spam.csv', encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

    x = df['v2']
    y = df['label']

    cv = CountVectorizer()

    x = cv.fit_transform(x)

    xtrain, xtest, ytrain, ytest =  train_test_split(x, y, test_size = 0.3, random_state=42)

    clf = MultinomialNB()
    clf.fit(xtrain, ytrain)
    clf.score(xtest, ytest)

    # ypred = clf.predict(xtest)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    
    return render_template('result.html', prediction=my_prediction)

    # print(classification_report(ytest, ypred))

    # joblib.dump(clf, 'model/spam.pkl')

if __name__ == '__main__':
    app.run(debug=True)
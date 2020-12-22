# import necessary libraries
import os
import datetime
import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

# Imports the method used for connecting to DBs
from sqlalchemy import create_engine
from sqlalchemy import desc

# Imports the methods needed to abstract classes into tables
from sqlalchemy.ext.declarative import declarative_base

# Allow us to declare column types
from sqlalchemy import Column, Integer, String, Float

from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)

from flask_sqlalchemy import SQLAlchemy

import sqlite3

# import nltk.classify.util
# from nltk.classify import NaiveBayesClassifier
# from nltk.corpus import movie_reviews
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.corpus import wordnet
# import nltk
# nltk.download('punkt')
# import nltk
# nltk.download('averaged_perceptron_tagger')
# import nltk
# nltk.download('tagsets')
# import nltk
# nltk.download('wordnet')
# nltk.download('movie_reviews')
# nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import emoji 
## Naive Bayes ##

#this is how Naive Bayes classifier expects the input
# def create_word_features(words):
#     useful_words = [word for word in words if word not in stopwords.words("english")]
#     my_dict = dict([(word, True) for word in useful_words])
#     return my_dict 

# neg_reviews = []
# for fileid in movie_reviews.fileids('neg'):
#     words = movie_reviews.words(fileid)
#     neg_reviews.append((create_word_features(words),"negative"))



# pos_reviews = []
# for fileid in movie_reviews.fileids('pos'):
#     words = movie_reviews.words(fileid)
#     pos_reviews.append((create_word_features(words),"positive"))

# train_set = neg_reviews[:750] +  pos_reviews[:750]
# test_set = neg_reviews[750:] + pos_reviews[750:]

# nbc = NaiveBayesClassifier.train(train_set)


# Import data for ml model
ml_file_path = 'static/Resources/ml_app_df.csv'
ml_input_df = pd.read_csv(ml_file_path)

# Create object with review data
df_x = ml_input_df['cleaned']

# Create object with class(star) data
df_y = ml_input_df['class']

# print(df_y)

# Split data into training and testing
X_train,X_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=41)

# Initialize tfidf vectorizer
tfidf = TfidfVectorizer(min_df=1)

# Fit training data to vectorizer
X = tfidf.fit_transform(X_train)

# Initialize Niave Bayes object
mnb = MultinomialNB()

# Cast y_train as int
y_train = y_train.astype('int')

# Fit Naive Bayes model
mnb.fit(X, y_train)

# Transform X_test
X_test = tfidf.transform(X_test)

# Initialize predict object for testing model accuracy
pred = mnb.predict(X_test)

# Initialize y_test object for testing model accuracy
actual = np.array(y_test)



# DATABASE_URL will contain the database connection string:
# app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///reviews.sqlite"

project_dir = os.path.dirname(os.path.abspath(__file__))
database_file = "sqlite:///{}".format(os.path.join(project_dir, "reviews.db"))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = database_file

app.config['SQLALCHEMY_TRACK_MODIFICATION'] = False

# Connects to the database using the app config
db = SQLAlchemy(app)

class Review(db.Model):
    id = db.Column(db.Integer, unique=True, primary_key=True)
    review = db.Column(db.String(155), unique=False, nullable=False)
    prediction = db.Column(db.String(155), nullable=False)

    def __repr__(self):
        return "<{}:{}:{}>".format(id, self.review, self.prediction)

@app.route("/")
def home():
    engine = create_engine("sqlite:///reviews.sqlite")

    table = engine.execute("SELECT * FROM reviews").fetchall()
    table = Review.query.all()

    return render_template("index.html", table=table)

@app.route("/send", methods=["GET", "POST"])
def send():
    engine = create_engine("sqlite:///reviews.sqlite")

    session = Session(engine)
    if request.method == "POST":
       
        # dash_input = request.form.get("review")
        # words = word_tokenize(dash_input)
        # words = create_word_features(words)

        # user_prediction = nbc.classify(words)

        #  Assign text area input to a variable
        dash_input = [request.form.get("review")]

        # Transform input
        dash_input = tfidf.transform(dash_input)
        
        # User sentiment prediction for naive bayes
        user_prediction = mnb.predict(dash_input)
        if user_prediction == 1:
            user_prediction = 'postive ' + emoji.emojize(":grinning_face_with_big_eyes:",use_aliases=True)
        else:
            user_prediction = 'negative ' + emoji.emojize(":disappointed:",use_aliases=True)

        review = Review(review=request.form.get("review"), prediction=user_prediction)
        db.session.add(review)
   
        db.session.commit()
        return redirect("/#after", code=302)

    table = Review.query.all()

    return render_template("form.html", table=table)

@app.route("/update", methods=["POST"])
def update():
    newreview = request.form.get("newreview")
    oldreview = request.form.get("oldreview")
    newpred = request.form.get("newpred")
    oldpred = request.form.get("oldpred")

# Assign text area input to a variable
    dash_input = [newreview]

    # words = word_tokenize(dash_input)
    # words = create_word_features(words)

    # user_prediction = nbc.classify(words)

    # Transform input
    dash_input = tfidf.transform(dash_input)

    # User sentiment prediction for naive bayes
    user_prediction = mnb.predict(dash_input)

    if user_prediction == 1:
        newpred = 'postive ' + emoji.emojize(":grinning_face_with_big_eyes:",use_aliases=True)
    else:
        newpred = 'negative ' + emoji.emojize(":disappointed:",use_aliases=True)

    review = Review.query.filter_by(review=oldreview).first()
    prediction = Review.query.filter_by(prediction=oldpred).first()

    review.review = newreview
    review.prediction = newpred
    db.session.commit()
    return redirect("/#after", code=302)

@app.route("/delete", methods=["POST"])
def delete():
    review = request.form.get("review")
    review = Review.query.filter_by(review=review).first()
    db.session.delete(review)
    db.session.commit()
    return redirect("/#after", code=302)

if __name__ == "__main__":
    app.run()


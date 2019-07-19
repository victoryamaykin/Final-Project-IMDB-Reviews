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
# import nltk
# nltk.download('movie_reviews')

# import necessary libraries
import os

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

app = Flask(__name__)

# DATABASE_URL will contain the database connection string:
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///reviews.sqlite"

# Connects to the database using the app config
db = SQLAlchemy(app)

engine = create_engine("sqlite:///reviews.sqlite")
# c = engine.connect()

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(db.engine, reflect=True)

Base.metadata.create_all(engine)

class Review(db.Model):
    __tablename__ = 'reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.String(255))
    # prediction = db.Column(db.String(255))

# Save references to each table
Reviews = Base.classes.reviews

# prepare session to receive user inputs
session = Session(engine)

@app.before_first_request
def setup():
    # Recreate database each time for demo
    db.drop_all()
    db.create_all()

# create route that renders index.html template
@app.route("/")
def home():

    table = engine.execute("SELECT * FROM reviews").fetchall()
    print(type(table))
    return render_template("index.html", table=table)

@app.route("/reviews")
def reviews():
    # Use Pandas to perform the sql query
    stmt = db.session.query(Reviews).statement
    df = pd.read_sql_query(stmt, db.session.bind)

    # Return a list of the reviews
    return jsonify(df)


    #this is how Naive Bayes classifier expects the input
# def create_word_features(words):
#         #Join all review words
#         results = engine.execute("SELECT * FROM reviews").fetchall()

#         text = [result[1] for result in results]
#         print(text)
#         useful_words = [word for word in words if word not in stopwords.words("english")]
#         my_dict = dict([(word, True) for word in useful_words])
#         return my_dict 

#         neg_reviews = []
#         for fileid in movie_reviews.fileids('neg'):
#             words = movie_reviews.words(fileid)
#             neg_reviews.append((create_word_features(words),"negative"))
#         print(neg_reviews[0])
#         print(len(neg_reviews))

#         pos_reviews = []
#         for fileid in movie_reviews.fileids('pos'):
#             words = movie_reviews.words(fileid)
#             pos_reviews.append((create_word_features(words),"positive"))
#         #print(pos_reviews[0])
#         print(len(pos_reviews))

#         train_set = neg_reviews[:750] +  pos_reviews[:750]
#         test_set = neg_reviews[750:] + pos_reviews[750:]
#         print(len(train_set), len(test_set))

#         #train the classifier
#         classifier = NaiveBayesClassifier.train(train_set)

#         #find accuracy percentage
#         accuracy = nltk.classify.util.accuracy(classifier,test_set)
#         print(accuracy * 100)

#         words = word_tokenize(text)
#         words = create_word_features(words)

#         results = classifier.classify(words)

#         print(results)
#         predictions = [result for result in results]
#         print(predictions)
#         df['prediction'] = predictions

#         return jsonify(df)

# Query the database and send the jsonified results
@app.route("/send", methods=["GET", "POST"])
def send():
    if request.method == "POST":
        review = request.form["review"]
        review = Review(review=review)
        db.session.add(review)
        db.session.commit()
        return redirect("/", code=302)

    return render_template("form.html")

if __name__ == "__main__":
    app.run()


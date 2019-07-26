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
    id = db.Column(db.Integer, unique=True, nullable=False, primary_key=True)
    review = db.Column(db.String(80), unique=True, nullable=False, primary_key=False)
    def __repr__(self):
        return "<Review: {}>".format(self.review)

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
        review = Review(review=request.form.get("review"))
        db.session.add(review)
        db.session.commit()
        return redirect("/", code=302)

    table = Review.query.all()

    return render_template("form.html", table=table)

@app.route("/update", methods=["POST"])
def update():
    newreview = request.form.get("newreview")
    oldreview = request.form.get("oldreview")
    review = Review.query.filter_by(review=oldreview).first()
    review.review = newreview
    db.session.commit()
    return redirect("/", code=302)

@app.route("/delete", methods=["POST"])
def delete():
    review = request.form.get("review")
    review = Review.query.filter_by(review=review).first()
    db.session.delete(review)
    db.session.commit()
    return redirect("/", code=302)

if __name__ == "__main__":
    app.run()


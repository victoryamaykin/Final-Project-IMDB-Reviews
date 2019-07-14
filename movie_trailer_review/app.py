#sklearn and ML
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
# from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import nltk

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
import datetime 
from sqlalchemy import Column, Integer, String, Float, DateTime

from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)

from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# DATABASE_URL will contain the database connection string:
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db/reviews.sqlite"

# Connects to the database using the app config
db = SQLAlchemy(app)

engine = create_engine("sqlite:///db/reviews.sqlite")
conn = engine.connect()

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(db.engine, reflect=True)

# prepare session to receive user inputs
session = Session(bind=engine)

# substatianate a class for columns
class Review(db.Model):
    __tablename__ = 'reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.String(255))
    # date = db.Column(db.String(255))

# Save references to each table
Reviews = Base.classes.reviews

@app.before_first_request
def setup():
    # Recreate database each time for demo
    # db.drop_all()
    db.create_all()

# create route that renders index.html template
@app.route("/")
def home():

    table = engine.execute("SELECT * FROM reviews").fetchall()
    
    return render_template("index.html", table=table)

@app.route("/reviews")
def reviews():
    # Use Pandas to perform the sql query
    stmt = db.session.query(Reviews).statement
    df = pd.read_sql_query(stmt, db.session.bind)

    # Return a list of the reviews
    return jsonify(list(df["review"]))

# Query the database and send the jsonified results
@app.route("/send", methods=["GET", "POST"])
def send():
    if request.method == "POST":
        review = request.form["review"]
        date = request.form["date"]

        review = Review(review=review)
        db.session.add(review)
        db.session.commit()
        return redirect("/", code=302)

    return render_template("form.html")


@app.route("/api/reviews")
def clean_review():
    
     #Join all review words
    results = engine.execute("SELECT * FROM reviews").fetchall()

    # id = [result[0] for result in results]
    text = [result[1] for result in results]
    # sentiment = [result[2] for result in results]
    return jsonify(text)

    #Tokenization of text
    import nltk
    nltk.download('stopwords')
    tokenizer=ToktokTokenizer()
    #Setting English stopwords
    stopword_list=nltk.corpus.stopwords.words('english')
   
    def remove_special_characters(text, remove_digits=True):
        pattern=r'[^a-zA-z0-9\s]'
        text=re.sub(pattern,'',text)
        return text
        #Apply function on review column
        text=text.apply(remove_special_characters)
    
    #removing the stopwords
    def remove_stopwords(text, is_lower_case=False):
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
            filtered_text = ' '.join(filtered_tokens)    
        return filtered_text

        #Apply function on review column
        text=text.apply(remove_stopwords)
    
    print(text)

    return render_template("index.html", text=text)

if __name__ == "__main__":
    app.run()

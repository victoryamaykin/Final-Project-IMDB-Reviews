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


import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# import necessary libraries
import os
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)

import sqlalchemy
from sqlalchemy.ext.automap import automap_base

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Session

# Imports the method used for connecting to DBs
from sqlalchemy import create_engine

# Imports the methods needed to abstract classes into tables
from sqlalchemy.ext.declarative import declarative_base

# Allow us to declare column types
from sqlalchemy import Column, Integer, String, Float 

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db/reviews.sqlite"
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '')

db = SQLAlchemy(app)

engine = create_engine("sqlite:///db/reviews.sqlite")
conn = engine.connect()
# Base.metadata.create_all(engine)

session = Session(bind=engine)

results = engine.execute("SELECT * FROM reviews").fetchall()
print(results) 

class Review(db.Model):
    __tablename__ = 'reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.String(255))

@app.before_first_request
def setup():
    # Recreate database each time for demo
    db.drop_all()
    db.create_all()

# create route that renders index.html template
@app.route("/")
def home():

    table = engine.execute("SELECT * FROM reviews").fetchall()
    
    return render_template("index.html", table=table)


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




@app.route("/show/reviews")
def show_review():

    from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
    from keras.layers import Bidirectional, GlobalMaxPool1D
    from keras.models import Model, Sequential
    from keras.layers import Convolution1D
    from keras import initializers, regularizers, constraints, optimizers, layers
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    maxlen = 130


    from nltk.stem import LancasterStemmer,WordNetLemmatizer
    from tensorflow.keras.models import load_model
    model = load_model("naive_bayes.h5")
    model.summary()
    
    print("model successfully loaded")

     #Join all review words
    results = engine.execute("SELECT * FROM reviews").fetchall()
    # print(results)

    lst_data = []
    id_count = 0
    for result in results:
         
        review_len=int(len(result[1])/2)
        id_count=id_count+1

        lst_data.append({
            # "id": "id" + str(id_count),
            "id": str(review_len)+"_"+str(id_count),
            # "id": "12311_10",
            "review": result[1]
        })

    reviews_df = pd.DataFrame(lst_data)
    orig_reviews_df = pd.DataFrame(lst_data)
    # print(reviews_df)



    stop_words = set(stopwords.words("english")) 
    lemmatizer = WordNetLemmatizer()

    csv_path = "D:/DataRepository/final_project/Final-Project-IMDB-Reviews/labeledTrainData.tsv"
    df = pd.read_csv(csv_path, low_memory=False, sep = '\t')
    df = df.drop(['id'], axis=1)

    df  = df[['review','sentiment']]    

    stop_words = set(stopwords.words("english")) 
    lemmatizer = WordNetLemmatizer()

    max_features = 6000
    tokenizer = Tokenizer(num_words=max_features)
    

    def clean_text(text):
        text = re.sub(r'[^\w\s]','',text, re.UNICODE)
        text = text.lower()
        text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
        text = [lemmatizer.lemmatize(token, "v") for token in text]
        text = [word for word in text if not word in stop_words]
        text = " ".join(text)
        return text

    df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))
    tokenizer.fit_on_texts(df['Processed_Reviews'])


#######################################################################
#######################################################################
#######################################################################
    # csv_path = "D:/DataRepository/final_project/Final-Project-IMDB-Reviews/labeledTrainData.tsv"
    # # csv_path = "D:/DataRepository/final_project/Final-Project-IMDB-Reviews/app_labeledTrainData.tsv"

    # # Import the CSV into a pandas DataFrame
    # df = pd.read_csv(csv_path, low_memory=False, sep = '\t')
    # df = df.drop(['id'], axis=1)

    # df  = df[['review','sentiment']]    

    # stop_words = set(stopwords.words("english")) 
    # lemmatizer = WordNetLemmatizer()

    # def clean_text(text):
    #     text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    #     text = text.lower()
    #     text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    #     text = [lemmatizer.lemmatize(token, "v") for token in text]
    #     text = [word for word in text if not word in stop_words]
    #     text = " ".join(text)
    #     return text

    # df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))


    # df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()

    # from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
    # from keras.layers import Bidirectional, GlobalMaxPool1D
    # from keras.models import Model, Sequential
    # from keras.layers import Convolution1D
    # from keras import initializers, regularizers, constraints, optimizers, layers
    # from keras.preprocessing.text import Tokenizer
    # from keras.preprocessing.sequence import pad_sequences

    # max_features = 6000
    # tokenizer = Tokenizer(num_words=max_features)
    # tokenizer.fit_on_texts(df['Processed_Reviews'])
    # list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])

    # maxlen = 130
    # X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    # y = df['sentiment']

    # embed_size = 128
    # model = Sequential()
    # model.add(Embedding(max_features, embed_size))
    # model.add(Bidirectional(LSTM(32, return_sequences = True)))
    # model.add(GlobalMaxPool1D())
    # model.add(Dense(20, activation="relu"))
    # model.add(Dropout(0.05))
    # model.add(Dense(1, activation="sigmoid"))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # batch_size = 100
    # epochs = 3
    # model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    
    # print(reviews_df)

#######################################################################
#######################################################################
#######################################################################



    reviews_df["review"]=reviews_df.review.apply(lambda x: clean_text(x))
    # reviews_df["sentiment"] = reviews_df["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
    # y_test = reviews_df["sentiment"]
    list_sentences_test = reviews_df["review"]
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
    print(X_te)
    prediction = model.predict(X_te)
    y_pred = (prediction > 0.5)
    from sklearn.metrics import f1_score, confusion_matrix
    # print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
    # print('Confusion matrix:')
    # confusion_matrix(y_pred, y_test)

    # print(y_pred)

    y_pred_df = pd.DataFrame(y_pred)

    review_sentiment_df=y_pred_df.replace({
    False:'Negative Review',
    True:'Positive Review'
    })

    review_sentiment_df = review_sentiment_df.rename(columns={0: 'Review_Sentiment'})



    print(review_sentiment_df)

    merge_reviews_df = orig_reviews_df.reset_index()
    merge_sentiment_df = review_sentiment_df.reset_index()

    # print(merge_reviews_df)
    # print(merge_sentiment_df)

    merged_output_df = pd.merge(merge_reviews_df, merge_sentiment_df, how='inner', on='index')
    
    # print(merged_output_df)
    merged_output_df = merged_output_df[["review","Review_Sentiment"]]
    print(merged_output_df)
    output_lst = merged_output_df.values.tolist()

    print(output_lst)

    json_list = []
    id_count = 0
    for x in range(len(output_lst)):

        id_count=id_count+1

        json_list.append({
            # "id": "id" + str(id_count),
            "id": str(id_count),
            "review": output_lst[x][0],
            "review_sentiment": output_lst[x][1]
        })

    print(json_list)


    return jsonify(json_list)

    # return jsonify(lst_data)


@app.route("/api/reviews")
def clean_review():
    
     #Join all review words
    results = engine.execute("SELECT * FROM reviews").fetchall()

    # id = [result[0] for result in results]
    review_words = [result[0] for result in results]
    # sentiment = [result[2] for result in results]
    return jsonify(review_words)

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
        review_words=review_words.apply(remove_special_characters)
    
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
        review_words=review_words.apply(remove_stopwords)
    
    print(review_words)

    return render_template("index.html", review_words=review_words)

if __name__ == "__main__":
    app.run()
# import necessary libraries
import os
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)


app = Flask(__name__)

from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db/reviews.sqlite"
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '')

db = SQLAlchemy(app)

from .models import Pet

@app.before_first_request
def setup():
    # Recreate database each time for demo
    # db.drop_all()
    db.create_all()

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")


# Query the database and send the jsonified results
@app.route("/send", methods=["GET", "POST"])
def send():
    if request.method == "POST":
        name = request.form["petName"]
        pet_type = request.form["petType"]
        age = request.form["petAge"]

        pet = Pet(name=name, type=pet_type, age=age)
        db.session.add(pet)
        db.session.commit()
        return redirect("/", code=302)

    return render_template("form.html")



@app.route("/api/reviews")
def pals():
    results = db.session.query(Pet.type, func.count(Pet.type)).group_by(Pet.type).all()

    pet_type = [result[0] for result in results]
    age = [result[1] for result in results]

    trace = {
        "x": pet_type,
        "y": age,
        "type": "bar"
    }

    return jsonify(trace)

    #Define stopwords
    stopwords = ['a', 'bad', 'good','br', 'film', 'movie', 'about', 'above', 'across', 'after', 'afterwards']

    #Join all review words
    review_words = db.session.query(Pet.type).all()

    # Generate a word cloud image based on bag of popcorn
    mask = np.array(Image.open("bag.jpg"))

    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10, max_font_size = 42,
                max_words=200,
                mask=mask).generate(review_words) 
  
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud, interpolation="bilinear") 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()

if __name__ == "__main__":
    app.run()

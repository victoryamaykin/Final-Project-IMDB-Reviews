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

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db.sqlite"
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '')

db = SQLAlchemy(app)

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(db.engine, reflect=True)

# Save references to each table
Reviews = Base.classes.reviews

# @app.before_first_request
# def setup():
#     # Recreate database each time for demo
#     # db.drop_all()
#     db.create_all()

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")


# Query the database and send the jsonified results
@app.route("/send", methods=["GET", "POST"])
def send():
    if request.method == "POST":
        first = request.form["first"]
        second = request.form["second"]
        third = request.form["third"]

        review = Reviews(first=first, second=second, third=third)
        db.session.add(review)
        db.session.commit()
        return redirect("/", code=302)

    return render_template("form.html")



@app.route("/api/reviews")
def wordcloud():

    #Define stopwords
    stopwords = ['and', 'bad', 'good','br', 'film', 'movie', 'about', 'above', 'across', 'after', 'afterwards']

    #Join all review words
    review_words = db.session.query(Reviews.first).all()

    # Generate a word cloud image based on bag of popcorn
    mask = np.array(Image.open("bag.jpg"))

    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10, 
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

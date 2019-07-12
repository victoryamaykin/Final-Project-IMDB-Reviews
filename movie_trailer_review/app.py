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



if __name__ == "__main__":
    app.run()

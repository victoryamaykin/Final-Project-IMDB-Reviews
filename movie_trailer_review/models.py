from .app import db


class Review(db.Model):
    __tablename__ = 'reviews'

    id = db.Column(db.Integer, primary_key=True)
    first = db.Column(db.String(64))
    second = db.Column(db.String(64))
    third = db.Column(db.String(64))

    def __repr__(self):
        return '<Review %r>' % (self.first)


from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20), unique=True, nullable=False)
    prediction = db.Column(db.JSON, nullable=False)

    def __repr__(self):
        return f'<Prediction {self.date}: {self.prediction}>'
    
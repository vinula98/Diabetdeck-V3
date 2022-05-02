import imp
from flask import Flask
from Model.predictor import predictDisease
import pickle


app = Flask(__name__)

# Members API Route
@app.route("/members")
def members():
    symptomInput ="Polyuria,Increased Appetite,Excessive Hunger"
    final_prediction = predictDisease(symptomInput)

    return print(final_prediction)

# # Model Output
# @app.route("/prediction")
# def prediction():
#     return{}


if __name__ == "__main__":
    app.run(debug=True)
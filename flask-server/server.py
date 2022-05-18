from urllib import response
from flask import Flask, request
from flask_cors import CORS, cross_origin
from Model.predictor import predictDisease

app = Flask(__name__)
CORS(app)

# Members API Route
@app.route("/predict", methods = ["POST"])
@cross_origin(origin='localhost')
def members():
    JsonReq = request.json
    JsonText = JsonReq['symptomData']

    response = predictDisease(JsonText)

    return {"data": response}


if __name__ == "__main__":
    app.run(debug=True)
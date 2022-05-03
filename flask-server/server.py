# from flask import Flask, request
from urllib import response
from flask import Flask, request
from flask_cors import CORS, cross_origin
from Model.predictor import predictDisease

app = Flask(__name__)
CORS(app)

# Members API Route
@app.route("/members", methods = ["POST"])
@cross_origin(origin='localhost')
def members():
    JsonReq = request.json
    JsonText = JsonReq['symptomData']

    response = predictDisease(JsonText)

    return {"data": response}

    # final_response = request.json 
    # symptomInput ="Polyuria,Increased Appetite,Excessive Hunger"
    # final_prediction = predictDisease(final_response)

    # return {"symptoms" : final_prediction}

# # Model Output
# @app.route("/prediction")
# def prediction():
#     return{}


if __name__ == "__main__":
    app.run(debug=True)
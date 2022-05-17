import pickle
from scipy.stats import mode
import numpy as np

final_svm_model = pickle.load(open('D:/Final Year Project/Diabetdeck-V3/flask-server/Model/save/svm/finalsvmmodel.h5', 'rb'))
final_lr_model = pickle.load(open('D:/Final Year Project/Diabetdeck-V3/flask-server/Model/save/lr/finallrmodel.h5', 'rb'))
final_rf_model = pickle.load(open('D:/Final Year Project/Diabetdeck-V3/flask-server/Model/save/rf/finalrfmodel.h5', 'rb'))
data_dict = pickle.load(open('D:/Final Year Project/Diabetdeck-V3/flask-server/Model/save/data_dictionary/datadictionary.h5', 'rb'))

# pickle.dump(data_dict, open('D:/Final Year Project/Diabetdeck-V3/flask-server/Model/save/data_dictionary/datadictionary.h5', 'wb'))

def predictDisease(symptoms):
    symptoms = symptoms.split(",")
     
    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
         
    # reshaping the input data and converting it into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
     
    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
     
    # making final prediction by taking Radom Forest model
    final_prediction = mode([rf_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
    }
    if final_prediction == 'Diabetes ':
        return ("You have Type 1 Diabetes. Diabetdeck strongly recommends that you visit your family doctor or an Endocrinologist.")
    else:
        return ("You do not have Diabetes. But if you have any other symptoms, Diabetdeck would strongly recommend that you visit a doctor as those symptoms could be of another disease.")
 
# Testing the function
# print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))
print(predictDisease("Polyuria,Increased Appetite,Excessive Hunger"))
from cgi import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import pickle
import requests
import json
import seaborn as sns

# Reading the train.csv by removing the last column since it's an empty column
DATA_PATH = "D:/Final Year Project/Diabetdeck-V3/flask-server/dataset/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis = 1)
 
# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

# Encoding the target value into numerical value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Splitting Data
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_training_data, X_testing_data, y_training_data, y_testing_data =train_test_split(
  X, y, test_size = 0.3, random_state = 24)

# Defining scoring metric for k-fold cross validation
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))
 
# Initializing Models
models = {
    "SVC":SVC(),
    "Logistic Regression":LogisticRegression(),
    "Random Forest":RandomForestClassifier(random_state=18)
}
 
# Producing cross validation score for the models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv = 10,
                             n_jobs = -1,
                             scoring = cv_scoring)

# Training and testing SVM Classifier
svm_model = SVC()
svm_model.fit(X_training_data, y_training_data)
preds = svm_model.predict(X_testing_data)
# pickle.dump(svm_model, open('model.pkl','wb'))
 
print(f"Accuracy of SVM Classifier\
: {accuracy_score(y_testing_data, preds)*100}")

cf_matrix = confusion_matrix(y_testing_data, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.show()

print("Precision Score : ", precision_score(y_testing_data, preds, average='macro'))

print("Recall Score : ", recall_score(y_testing_data, preds, average='macro'))

print("F1 Score : ", f1_score(y_testing_data, preds, average='macro'))

# Training and testing Logistic Regression
lr_model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear')
lr_model.fit(X_training_data, y_training_data)
lr_model.score(X_training_data, y_training_data)
preds = lr_model.predict(X_testing_data)
# pickle.dump(lr_model, open('model.pkl','wb'))
 
print(f"Accuracy of Logistic Regression\
: {accuracy_score(y_testing_data, preds)*100}")

cf_matrix = confusion_matrix(y_testing_data, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Logistic Regression on Test Data")
plt.show()

print("Precision Score : ", precision_score(y_testing_data, preds, average='macro'))

print("Recall Score : ", recall_score(y_testing_data, preds, average='macro'))

print("F1 Score : ", f1_score(y_testing_data, preds, average='macro'))
 
# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_training_data, y_training_data)
preds = rf_model.predict(X_testing_data)
# pickle.dump(rf_model, open('model.pkl','wb'))
 
print(f"Accuracy of Random Forest Classifier\
: {accuracy_score(y_testing_data, preds)*100}")

cf_matrix = confusion_matrix(y_testing_data, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.show()

print("Precision Score : ", precision_score(y_testing_data, preds, average='macro'))

print("Recall Score : ", recall_score(y_testing_data, preds, average='macro'))

print("F1 Score : ", f1_score(y_testing_data, preds, average='macro'))

# Training the models on all training data and validating on test data
final_svm_model = SVC()
final_lr_model = LogisticRegression()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
pickle.dump(final_svm_model, open('D:/Final Year Project/Diabetdeck-V3/flask-server/Model/save/svm/finalsvmmodel.h5', 'wb'))
final_lr_model.fit(X, y)
pickle.dump(final_lr_model, open('D:/Final Year Project/Diabetdeck-V3/flask-server/Model/save/lr/finallrmodel.h5', 'wb'))
final_rf_model.fit(X, y)
pickle.dump(final_rf_model, open('D:/Final Year Project/Diabetdeck-V3/flask-server/Model/save/rf/finalrfmodel.h5', 'wb'))
 
# Reading the test data
test_data = pd.read_csv("D:/Final Year Project/Diabetdeck-V3/flask-server/dataset/Testing.csv").dropna(axis=1)
 
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])

symptoms = X.columns.values
 
# Creating a symptom index dictionary to encode the input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}
pickle.dump(data_dict, open('D:/Final Year Project/Diabetdeck-V3/flask-server/Model/save/data_dictionary/datadictionary.h5', 'wb'))
 
# Function for final prediction
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
     
    # making final prediction by taking mode of all predictions
    final_prediction = mode([rf_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
    }
    if final_prediction == 'Diabetes ':
        return ("You have Type 1 Diabetes. Diabetdeck strongly recommends that you visit your family doctor or an Endocrinologist.")
    else:
        return ("You do not have Diabetes. But if you have any other symptoms, Diabetdeck would strongly recommend that you visit a doctor as those symptoms could be of another disease.")
 
# Testing the function
print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))
# print(predictDisease("Polyuria,Increased Appetite,Excessive Hunger"))
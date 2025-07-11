import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("encoded_data.csv")

x = df.drop(columns=["diabetes"])
y = df["diabetes"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=26)

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def compute_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

def train(x, y, epochs=1001, lr=0.001):
    weights = np.zeros(x.shape[1])
    for _ in range(epochs):
        z = np.dot(x, weights)
        y_pred = sigmoid(z)
        gradient = np.dot(x.T, (y_pred - y))/len(y)
        weights -= gradient*lr

        # if _ % 100 == 0:
        #     print(f"Epoch: {_}, Loss: {compute_loss(y, y_pred)}")
    return weights

def predict(x, weights, threshold = 0.54):
    z = np.dot(x, weights)
    y_pred_prob = sigmoid(z)
    return (y_pred_prob >= threshold).astype(int)

weights = train(x_train, y_train)

y_pred = predict(x_test, weights)

accuracy = np.mean(y_test == y_pred)

print(weights)

print("Accuracy", accuracy*100, "%")
print("F1 Score: ", f1_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

influences = np.zeros(6)

influence_statuses = {
    "age_status": "NA",
    "hypertension_status": "NA",
    "heart_disease_status": "NA",
    "bmi_status": "NA",
    "hba1c_status": "NA",
    "glucose_status": "NA",
}

keys = ["age_status", "hypertension_status", "heart_disease_status", "bmi_status", "hba1c_status", "glucose_status"]

user_age = 66
user_hypertension = 1
user_heart_disease = 1
user_bmi = 32.1
user_hba1c = 7.2
user_glucose = 174

bmi_lower_bound = 18.5
bmi_upper_bound = 24.9

#1. influence for age 
delta = (user_age - 30) / 30
age_influence = delta * weights[0] * 1000 * 0.5
age_influence = round(age_influence, 2)
influences[0] = age_influence

#2. influence for hypertension
if user_hypertension == 1:
    hypertension_influence = weights[1] * 1000 * 0.5
else: 
    hypertension_influence = 0
hypertension_influence = round(hypertension_influence, 2)
influences[1] = hypertension_influence

#3. influence for heart disease
if user_heart_disease == 1:
    heart_disease_influence = weights[2] * 1000 * 0.5
elif user_heart_disease == 0: 
    heart_disease_influence = 0
heart_disease_influence = round(heart_disease_influence, 2)
influences[2] = heart_disease_influence

#4. influence for bmi
if user_bmi > bmi_upper_bound:
    alpha = (user_bmi - bmi_upper_bound) / bmi_upper_bound
    bmi_influence = weights[3] * alpha * 1000

elif user_bmi < bmi_lower_bound:
    alpha = (bmi_lower_bound - user_bmi) / bmi_lower_bound
    bmi_influence = weights[3] * alpha * 1000

else: 
    bmi_influence = 0
bmi_influence = round(bmi_influence, 2)
influences[3] = bmi_influence

#5. influence for blood glucose levels
if user_glucose > 126:
    beta = (user_glucose - 100)/ 126
    glucose_influence = weights[5] * beta * 1000

elif user_glucose > 100 and user_glucose <= 126:
    beta = (user_glucose - 100) / 126
    glucose_influence = weights[5] * beta * 0.8 * 1000

elif user_glucose <= 70:
    beta = (70 - user_glucose) / 70
    glucose_influence = weights[5] * beta * 1000
glucose_influence = round(glucose_influence, 2)
influences[4] = glucose_influence

#6. influence for hba1c levels
if user_hba1c > 6.4:
    charlie = (user_hba1c - 5.7) / 6.4
    hba1c_influence = weights[4] * charlie * 1000

elif user_hba1c >= 5.7 and user_hba1c <= 6.4:
    charlie = (user_hba1c - 5.7) / 6.4
    hba1c_influence = weights[4] * charlie * 0.8 * 1000

elif user_hba1c < 5.7:
    hba1c_influence = 0
hba1c_influence = round(hba1c_influence, 2)
influences[5] = hba1c_influence


total_influence = age_influence + glucose_influence + bmi_influence + hba1c_influence
age_influence_percentage = age_influence/total_influence * 100
glucose_influence_percentage = glucose_influence/total_influence * 100
bmi_influence_percentage = bmi_influence/total_influence * 100
hba1c_influence_percentage = hba1c_influence/total_influence * 100

for i in range(len(influences)):
    if influences[i] > 40:
        status = "Extreme"
    elif influences[i] > 30:
        status = "High"
    elif influences[i] > 20: 
        status = "Moderate"
    elif influences[i] > 10:
        status = "Low"
    else:
        status = "Little to None"
    influence_statuses[keys[i]] = status
    

print(f"Age Weight: {weights[0]}, Hypertension Weight: {weights[1]}, Heart Disease Weight: {weights[2]}, BMI Weight: {weights[3]}, HBA1C Weight: {weights[4]}, Glucose Weight: {weights[5]}")

print(f"Age Influence: {influences[0]}, Hypertension Influence: {hypertension_influence}, Heart Disease Influence: {heart_disease_influence}, BMI Influence: {bmi_influence}, HBA1C Influence: {hba1c_influence}, Glucose influence: {glucose_influence}")

print(influence_statuses)
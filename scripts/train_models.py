import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logistic_regression_model import LogisticRegression
from logistic_regression_model import ImprovedLogisticRegression

import joblib

df = pd.read_csv("data/encoded_data.csv")

x = df.drop(columns=["diabetes"])
y = df["diabetes"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=26)

model = LogisticRegression()
model1 = ImprovedLogisticRegression()


weights = model.train(x_train, y_train)
# LogisticRegression.plot_loss(model.loss_history)

joblib.dump(model, 'models/lr_model.pkl')

y_pred = model.predict(x_test)

accuracy = np.mean(y_test == y_pred)

print("Accuracy", accuracy*100, "%")
print("F1 Score:", f1_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# lambdas = [0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06]
# thresholds = [0.57, 0.55, 0.53, 0.5]

# best_f1 = 0
# best_prec = 0
# best_lambda = None
# best_reg = None
# best_threshold = None

# for reg in ['none', 'l1', 'l2']:
#     for lam in lambdas:
#         model = LogisticRegression(lambda_=lam, regularisation=reg)
#         model.train(x_train, y_train)

#         for thresh in thresholds:
#             y_pred = model.predict(x_test, threshold=thresh)
#             f1_score_ = f1_score(y_test, y_pred)
#             prec = precision_score(y_test, y_pred)

#             if prec >= 0.94 and f1_score_ > best_f1:
#                 best_prec = prec
#                 best_f1 = f1_score_
#                 best_threshold = thresh
#                 best_lambda = lam
#                 best_reg = reg

# print(f"Best Lambda: {best_lambda}, Best Reg: {best_reg}, Best Prec: {best_prec}, Best Threshold: {best_threshold}, Best F1 Score: {best_f1}")

# Best Lambda: 0.1, Best Reg: l2, Best Prec: 0.9687034277198212, Best Threshold: 0.6, Best F1 Score: 0.5517826825127334

# for t in [0.5, 0.4, 0.3, 0.2]:
#     preds = model.predict(x_test, threshold=t)
#     print(f"Threshold: {t}")
#     print("Accuracy", accuracy*100, "%")
#     print("F1 Score:", f1_score(y_test, preds))
#     print("Recall:", recall_score(y_test, preds))
#     print("Precision:", precision_score(y_test, preds))
#     print(confusion_matrix(y_test, preds))
#     print("-" * 30)

# lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
# thresholds = np.arange(0.1, 0.6, 0.05)

# best_f1 = 0
# best_lambda = None
# best_threshold = None

# lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
# thresholds = np.arange(0.1, 0.6, 0.05)

# best_f1 = 0
# best_lambda = None
# best_threshold = None

# for lam in lambdas:
#     model = LogisticRegression(lambda_=lam)
#     model.train(x_train, y_train)

#     for thresh in thresholds:
#         y_pred = model.predict(x_test, threshold=thresh)
#         f1 = f1_score(y_test, y_pred)
#         prec = precision_score(y_test, y_pred)

#         # Only consider models with precision >= 92%
#         if prec >= 0.957 and f1 > best_f1:
#             best_f1 = f1
#             best_lambda = lam
#             best_threshold = thresh

# print(f"Best F1: {best_f1} with lambda={best_lambda} and threshold={best_threshold}")


# user_age = float(input("What is your age? Enter here: "))
# user_hypertension = float(input("Do you have high blood pressure? (1 for yes, 0 for no) Enter here: "))
# user_heart_disease = float(input("Do you or have you had any heart diseases?(1 for yes, 0 for no) Enter here: "))
# user_bmi = float(input("What is your BMI? Enter here: "))
# user_hba1c = float(input("What are your hba1c levels? Enter here: "))
# user_glucose = float(input("What are your blood glucose levels? Enter here: "))

# use these variables to check functionality without inputting data

user_age = 66
user_hypertension = 1
user_heart_disease = 1
user_bmi = 32.1
user_hba1c = 7.2
user_glucose = 174

influences = np.zeros(6)

influence_statuses = {
    "age_status": "NA",
    "hypertension_status": "NA",
    "heart_disease_status": "NA",
    "bmi_status": "NA",
    "hba1c_status": "NA",
    "glucose_status": "NA",
}

influence_scores = {
    "age_score": 0,
    "hypertension_score": 0,
    "heart_disease_score": 0,
    "bmi_score": 0,
    "hba1c_score": 0,
    "glucose_score": 0,
}

status_keys = ["age_status", "hypertension_status", "heart_disease_status", "bmi_status", "hba1c_status", "glucose_status"]

score_keys = ["age_score", "hypertension_score", "heart_disease_score", "bmi_score", "hba1c_score", "glucose_score"]


bmi_lower_bound = 18.5
bmi_upper_bound = 24.9

#1. influence for age 
delta = (user_age - 30) / 30
age_influence = delta * model.weights[0] * 1000 * 0.5
age_influence = round(age_influence, 2)
influences[0] = age_influence

#2. influence for hypertension
if user_hypertension == 1:
    hypertension_influence = model.weights[1] * 1000 * 0.5
else: 
    hypertension_influence = 0
hypertension_influence = round(hypertension_influence, 2)
influences[1] = hypertension_influence

#3. influence for heart disease
if user_heart_disease == 1:
    heart_disease_influence = model.weights[2] * 1000 * 0.5
elif user_heart_disease == 0: 
    heart_disease_influence = 0
heart_disease_influence = round(heart_disease_influence, 2)
influences[2] = heart_disease_influence

#4. influence for bmi
if user_bmi > bmi_upper_bound:
    alpha = (user_bmi - bmi_upper_bound) / bmi_upper_bound
    bmi_influence = model.weights[3] * alpha * 1000

elif user_bmi < bmi_lower_bound:
    alpha = (bmi_lower_bound - user_bmi) / bmi_lower_bound
    bmi_influence = model.weights[3] * alpha * 1000

else: 
    bmi_influence = 0
bmi_influence = round(bmi_influence, 2)
influences[3] = bmi_influence

#5. influence for blood glucose levels
if user_glucose > 126:
    beta = (user_glucose - 100)/ 126
    glucose_influence = model.weights[5] * beta * 1000

elif user_glucose > 100 and user_glucose <= 126:
    beta = (user_glucose - 100) / 126
    glucose_influence = model.weights[5] * beta * 0.8 * 1000

elif user_glucose > 70 and user_glucose <= 100:
        glucose_influence = 0

elif user_glucose <= 70:
    beta = (70 - user_glucose) / 70
    glucose_influence = model.weights[5] * beta * 1000
glucose_influence = round(glucose_influence, 2)
influences[5] = glucose_influence

#6. influence for hba1c levels
if user_hba1c > 6.4:
    charlie = (user_hba1c - 5.7) / 6.4
    hba1c_influence = model.weights[4] * charlie * 1000

elif user_hba1c >= 5.7 and user_hba1c <= 6.4:
    charlie = (user_hba1c - 5.7) / 6.4
    hba1c_influence = model.weights[4] * charlie * 0.8 * 1000

elif user_hba1c < 5.7:
    hba1c_influence = 0
hba1c_influence = round(hba1c_influence, 2)
influences[4] = hba1c_influence


total_influence = age_influence + glucose_influence + bmi_influence + hba1c_influence + heart_disease_influence + hypertension_influence
age_influence_percentage = age_influence/total_influence * 100
glucose_influence_percentage = glucose_influence/total_influence * 100
bmi_influence_percentage = bmi_influence/total_influence * 100
hba1c_influence_percentage = hba1c_influence/total_influence * 100
heart_disease_percentage = heart_disease_influence/total_influence * 100
hypertension_percentage = hypertension_influence/total_influence * 100

for i in range(len(influences)):
    if influences[i] > 45:
        status = "Extreme"
        score = "10/10"
    elif influences[i] > 40:
        status = "Extreme"
        score = "9/10"
    elif influences[i] > 35:
        status = "High"
        score = "8/10"
    elif influences[i] > 30:
        status = "High"
        score = "7/10"
    elif influences[i] > 25: 
        status = "Moderate"
        score = "6/10"
    elif influences[i] > 20: 
        status = "Moderate"
        score = "5/10"
    elif influences[i] > 15:
        status = "Moderate"
        score = "4/10"
    elif influences[i] >= 10:
        status = "Moderate/Low"
        score = "3/10"
    elif influences[i] >= 6 and influences[i] < 10:
        status = "Low"
        score = "2/10"
    elif influences[i] >= 3 and influences[i] < 6:
        status = "Low"
        score = "1/10"
    else:
        status = "No"
        score = "0/10"
    influence_statuses[status_keys[i]] = status
    influence_scores[score_keys[i]] = score
    

print("")

# print(f"Age Weight: {model.weights[0]}, Hypertension Weight: {model.weights[1]}, Heart Disease Weight: {model.weights[2]}, BMI Weight: {model.weights[3]}, HBA1C Weight: {model.weights[4]}, Glucose Weight: {model.weights[5]}")

# print(f"Age Influence: {influences[0]}, Hypertension Influence: {hypertension_influence}, Heart Disease Influence: {heart_disease_influence}, BMI Influence: {bmi_influence}, HBA1C Influence: {hba1c_influence}, Glucose influence: {glucose_influence}")

# print(influence_statuses)
# print(influence_scores)

# print(f"'Age' Affect: {influence_statuses['age_status']}, {influence_scores['age_score']} ")
# print(f"'High Blood Pressure' Affect: {influence_statuses['hypertension_status']}, {influence_scores['hypertension_score']} ")
# print(f"'Heart Disease' Affect: {influence_statuses['heart_disease_status']}, {influence_scores['heart_disease_score']} ")
# print(f"'BMI' Affect: {influence_statuses['bmi_status']}, {influence_scores['bmi_score']} ")
# print(f"'hba1c' Affect: {influence_statuses['hba1c_status']}, {influence_scores['hba1c_score']} ")
# print(f"'Blood Glucose Levels' Affect: {influence_statuses['glucose_status']}, {influence_scores['glucose_score']} ")

# if "Extreme" in influence_statuses.values():
#     print("")
#     print("You got issues fr twin you gotta fix that asap")
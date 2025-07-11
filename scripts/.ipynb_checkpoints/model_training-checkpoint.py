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

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def compute_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

def train(x, y, epochs=3000, lr=0.001):
    weights = np.zeros(x.shape[1])
    for _ in range(epochs):
        z = np.dot(x, weights)
        y_pred = sigmoid(z)
        gradient = np.dot(x.T, (y_pred - y))/len(y)
        weights -= gradient*lr

        if _ % 100 == 0:
            print(f"Epoch: {_}, Loss: {compute_loss(y, y_pred)}")
    return weights

def predict(x, weights, threshold = 0.65):
    z = np.dot(x, weights)
    y_pred_prob = sigmoid(z)
    return (y_pred_prob >= threshold).astype(int)

weights = train(x_train, y_train)

y_pred = predict(x_test, weights)

accuracy = np.mean(y_test == y_pred)

print("Accuracy", accuracy, "%")
print("F1 Score: ", f1_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
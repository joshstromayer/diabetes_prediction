import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib as plt

class LogisticRegression:
    def __init__(self, fit_intercept=True, lambda_=0.13, regularisation='none'):
        self.fit_intercept = fit_intercept
        self.scaler = StandardScaler()
        self.weights = None
        self.lambda_ = lambda_
        self.regularisation = regularisation
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1+np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        cross_entropy = -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

        if self.regularisation == 'L2':
            reg_loss = self.lambda_ * np.sum(self.weights[1:] ** 2)
        elif self.regularisation == 'L1':
            reg_loss = self.lambda_ * np.abs(self.weights[1:])
        else: 
            reg_loss = 0

        return cross_entropy + reg_loss

    def train(self, x, y, early_stopping=True, epochs=3001, patience=100, lr=0.01):
        x_scaled = self.scaler.fit_transform(x)
        
        best_loss = float('inf')
        patience_counter = 0
        tolerance = 1e-5

        if self.fit_intercept:
            x_with_intercept = np.column_stack([np.ones(x_scaled.shape[0]), x_scaled])
        else:
            x_with_intercept = x_scaled

        self.weights = np.random.normal(0, 0.01, x_with_intercept.shape[1])

        for _ in range(epochs):
            z = np.dot(x_with_intercept, self.weights)
            y_pred = self.sigmoid(z)

            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            gradient = np.dot(x_with_intercept.T, (y_pred - y))/len(y)
            
            if self.regularisation == 'L2': 
                gradient[1:] += 2 * self.lambda_ * self.weights[1:]
            elif self.regularisation == 'L1':
                gradient[1:] += self.lambda_ * np.sign(self.weights[1:])

            self.weights -= gradient*lr


            if early_stopping:
                if loss < best_loss - tolerance:
                    best_loss = loss
                    patience_counter = 0
                else: 
                    patience_counter += 1
                    if patience_counter > patience:
                        print(f"Early stopping at epoch {_}")
                        break

            if _ % 100 == 0:
                print(f"Epoch: {_}, Loss: {self.compute_loss(y, y_pred)}")


    def predict(self, x, threshold = 0.53):

        x_scaled = self.scaler.transform(x)
        

        if self.fit_intercept:
            x_with_intercept = np.column_stack([np.ones(x_scaled.shape[0]), x_scaled])
        else:
            x_with_intercept = x_scaled

        z = np.dot(x_with_intercept, self.weights)
        y_pred_prob = self.sigmoid(z)
        return (y_pred_prob >= threshold).astype(int)
    
    def plot_loss(loss_history):
        plt.figure(figsize=(10,6))
        plt.plot(loss_history, label='loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epoch')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from nn.train_test_split import *
from nn.models import Sequential
from nn.layers import Sigmoid, Softmax
from nn.metrics import accuracy


#PROCESS DATASET
dataset = pd.read_csv("data.csv", header=None)
dataset[1] = [1 if result == "M" else 0 for result in dataset[1]]

X = dataset.drop(columns=[1]).to_numpy()
y = dataset[1].to_numpy()
#SPLIT DATA
scaler = StandardScaler()
Xn = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xn, y, train_size=0.8)

net = Sequential(input_dim=X_train.shape[1], layers=[
    Sigmoid(64, name="layer1"),
    Sigmoid(32, name="layer2"),
    Sigmoid(24, name="layer3"),
    Sigmoid(12, name="layer4"),
    Sigmoid(2, name="layer5"),
    Softmax(2, name="output_layer")
])

net.load("model_data")

y_pred_onehot = net.predict(X_test)
y_pred = y_pred_onehot.argmax(axis=1)
for layer in net.layers:
    print(layer.weights.shape)
print(y_pred_onehot)
print(y_pred)
print(y_test)
acc = accuracy(y_pred, y_test)
print(f"Accuracy: {acc}%")
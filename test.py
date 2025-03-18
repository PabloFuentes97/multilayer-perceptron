import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from nn.train_test_split import *
from nn.models import Sequential
from nn.layers import Sigmoid, Softmax, Tanh
from nn.regularizers import L2
from nn.metrics import accuracy


#PROCESS DATASET
dataset = pd.read_csv("data.csv", header=None)
dataset[1] = [1 if result == "M" else 0 for result in dataset[1]]

X = dataset.drop(columns=[1]).to_numpy()
y = dataset[1].to_numpy()
#SPLIT DATA
classes = np.unique(y)
num_classes = len(classes)
scaler = StandardScaler()
Xn = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xn, y, train_size=0.8)
y_train = np.identity(n=num_classes)[y_train]

net = Sequential(input_dim=X_train.shape[1], layers=[
    Sigmoid(24, name="layer1"),
    Sigmoid(12, name="layer2"),
    Sigmoid(2, name="layer3"),
    Softmax(2, name="output_layer")
])
net.load("model_data")

predictions_onehot = net.predict(X_test)
predictions = predictions_onehot.argmax(axis=1)
acc = accuracy(predictions, y_test)
print(f"Accuracy: {acc}%")
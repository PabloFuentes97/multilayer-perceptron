import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from nn.metrics import accuracy
import sys
import joblib

#CHECK ARGS
args = sys.argv
if len(args) != 2:
    print("Bad number of arguments")
    exit(1)

filename = args[1]
#LOAD FILE TO DATAFRAME
try:
    dataset = pd.read_csv(filename, header=None)
except FileNotFoundError:
    print("File not found!")
    exit(2)

dataset[1] = [1 if result == "M" else 0 for result in dataset[1]]

X_test = dataset.drop(columns=[1]).to_numpy()
y_test = dataset[1].to_numpy()
#NORMALIZE DATA
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

net = joblib.load("model")
y_probs = net.predict(X_test)
y_pred = y_probs.argmax(axis=1)
acc = accuracy(y_pred, y_test)
print(f"Accuracy: {acc}%")
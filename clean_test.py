import numpy as np
import pandas as pd
from nn.models import *
from sklearn.preprocessing import StandardScaler
from nn.metrics import accuracy
from nn.loss import BinaryCrossEntropy
from nn.metrics import f1_score
import sys

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

X_test = dataset.drop(columns=[0, 10, 12, 15, 19, 20]).to_numpy()
y_test = dataset[1].to_numpy()
y_test_onehot = np.identity(2)[y_test]

#NORMALIZE DATA
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

#LOAD MODEL
net = load("my_model")

#PREDICT AND METRICS
y_probs = net.predict(X_test)
criterion = BinaryCrossEntropy()
loss = criterion(y_probs, y_test_onehot)
y_true_prob = y_probs[:, 1]
y_pred = y_probs.argmax(axis=1)

print(f"Loss: {loss}")
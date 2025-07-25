import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from nn.optimizers import *
from nn.models import *
from nn.layers import ReLU, Linear, Softmax
from nn.loss import BinaryCrossEntropy
import matplotlib.pyplot as plt
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

X = dataset.drop(columns=[0, 10, 12, 15, 19, 20]).to_numpy()
y = dataset[1].to_numpy()

#SPLIT DATA
classes = np.unique(y)
num_classes = len(classes)
scaler = StandardScaler()
Xn = scaler.fit_transform(X)
y = np.identity(n=num_classes)[y]
np.random.seed(42)

best_model = None
best_loss = np.inf
criterion = BinaryCrossEntropy()
optimizers = [
    SGD,
    SGDMomentum,
    RMSProp,
    Adam
]

optims_history = []
i = 0
for optim_type in optimizers:
    model = create_net_from_file("net_def.json")
    optim = optim_type(model)
    model.compile(criterion, optim)
    history = model.fit(Xn, y, epochs=100, batch_size=32, verbose=0)
    if history["train"]["loss"][-1] < best_loss:
        best_loss = history["train"]["loss"][-1]
        best_model = (i, model)
    optims_history.append(history)
    i += 1

print("Best model:", best_model[0])  
colors = ["red", "blue", "green", "pink"]

i = 0
for hist in optims_history:
    loss_history = hist["train"]["loss"]
    plt.plot(list(range(len(loss_history))), loss_history, color=colors[i], alpha=0.6)
    i += 1

plt.legend(labels=["SGD", "SGDMomentum", "RMSProp", "Adam"])
plt.show()
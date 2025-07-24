import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from nn.train_test_split import *
from nn.models import *
from nn.metrics import *
from nn.callbacks import EarlyStopping
import seaborn as sns
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
X_train, X_cv, y_train, y_cv = train_test_split(Xn, y, train_size=0.8)
y_train_bin = y_train
y_train = np.identity(n=num_classes)[y_train]
y_cv = np.identity(n=num_classes)[y_cv]

np.random.seed(42)

net = create_net_from_file("net_def.json")

early_stopping = EarlyStopping(net, min_delta=0.001, patience=20, verbose=True, restore_best_weights=True, start_from_epoch=5)

history = net.fit(X_train, y_train, epochs=200, batch_size=16, validation=True, validation_data=(X_cv, y_cv), verbose=0)

#SERIALIZE MODEL
save(net, "my_model")

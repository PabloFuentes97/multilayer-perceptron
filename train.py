import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from nn.train_test_split import *
from nn.models import Sequential
from nn.layers import Sigmoid, Softmax
from nn.loss import BinaryCrossEntropy, CategoricalCrossEntropy
from nn.optimizers import SGD, Adam, create_mini_batches
from nn.metrics import accuracy
from nn.regularizers import L2
from nn.callbacks import EarlyStopping
import seaborn as sns

#PROCESS DATASET
dataset = pd.read_csv("data.csv", header=None)
dataset[1] = [1 if result == "M" else 0 for result in dataset[1]]
print(dataset.describe())
vars = dataset.columns
vars = vars.delete(1)

X = dataset.drop(columns=[1]).to_numpy()
y = dataset[1].to_numpy()

#PLOT HISTOGRAMS
fig, ax = plt.subplots(4, 8)
ax = ax.flatten()
y_f = y.flatten()
idx_m = np.where(y_f == 1)
idx_b = np.where(y_f == 0)
n, m = X.shape
for i in range(m):
    X_i = X[:, i]
    X_m = X_i[idx_m]
    X_b = X_i[idx_b]
    ax[i].hist(X_m, alpha=0.5, color="red")
    ax[i].hist(X_b, alpha=0.5, color="blue")
    ax[i].set_title(f"{i}")

fig.legend(labels=["malign", "benign"])    
plt.show()

#SPLIT DATA
classes = np.unique(y)
num_classes = len(classes)
scaler = StandardScaler()
Xn = scaler.fit_transform(X)
X_train, X_, y_train, y_ = train_test_split(Xn, y, train_size=0.6)
y_train = np.identity(n=num_classes)[y_train]
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, train_size=0.5)
y_cv = np.identity(n=num_classes)[y_cv]

#MY MODEL
input_features = X_train.shape[1]
net = Sequential([
    Sigmoid(31, 24, regularizer=L2(0.5), name="layer1"),
    Sigmoid(24, 12, regularizer=L2(0.5), name="layer2"),
    Sigmoid(12, 2, regularizer=L2(0.5), name="layer3"),
    Softmax(2, 2, regularizer=L2(0.5), name="output_layer")
])

loss = CategoricalCrossEntropy(net)
optimizer = Adam(net, lr=0.01)

net.summary()
net.compile(loss, optimizer)

'''history = net.fit(X_train, y_train, epochs=100, batch_size=512, validation=True)
loss_history = history["loss"]
val_loss_history = history["val_loss"]'''

loss_history = []
val_loss_history = []
acc_history = []
val_acc_history = []

epochs = 100
batches_num = 15
batch_size = 256

early_stopping = EarlyStopping(net, min_delta=0.01, patience=5, verbose=True, restore_best_weights=True, start_from_epoch=5)

for epoch in range(epochs):
    #training step
    for batch_x, batch_y in create_mini_batches(X_train, y_train, batches_num, batch_size):
        y_pred = net.forward(batch_x)
        j = loss(y_pred, batch_y)
        loss.backward()
        optimizer.update(epoch)
        acc = accuracy(y_pred.argmax(axis=1), batch_y.argmax(axis=1))
    loss_history.append(j)
    acc_history.append(acc)
    
    #validation
    y_pred_val = net.forward(X_cv)
    j_val = loss(y_pred_val, y_cv)
    acc_val = accuracy(y_pred_val.argmax(axis=1), y_cv.argmax(axis=1))
    val_loss_history.append(j_val)
    val_acc_history.append(acc_val)
    
    #early_stopping
    if early_stopping(epoch, j):
        break
    

print("Loss after training:", loss_history[-1])
print("Validation Loss after training:", val_loss_history[-1])

#LOSS PLOT
plt.plot(list(range(len(loss_history))), loss_history, color="orange")
plt.plot(list(range(len(val_loss_history))), val_loss_history, color="blue", linestyle="dotted", alpha=0.8)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss history")
plt.show()

#ACC PLOT
'''
acc_history = history["acc"]
val_acc_history = history["val_acc"]'''
plt.plot(list(range(len(acc_history))), acc_history, color="orange")
plt.plot(list(range(len(val_acc_history))), val_acc_history, color="blue", linestyle="dotted", alpha=0.8)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy history")
plt.show()

predictions_onehot = net.predict(X_test)
#print("Predictions one-hot-coded:", predictions_onehot)
predictions = predictions_onehot.argmax(axis=1)
#print("Predictions:", predictions)
acc = accuracy(predictions, y_test)
print(f"Accuracy: {acc * 100}%")

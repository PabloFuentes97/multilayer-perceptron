import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from nn.train_test_split import *
from nn.models import Sequential
from nn.layers import Sigmoid, Softmax, ReLU, LeakyReLU, Tanh
from nn.loss import BinaryCrossEntropy, CategoricalCrossEntropy
from nn.optimizers import SGD, Adam, create_mini_batches
from nn.metrics import *
from nn.regularizers import L2
from nn.callbacks import EarlyStopping
import seaborn as sns

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
X_train, X_, y_train, y_ = train_test_split(Xn, y, train_size=0.6)
y_train = np.identity(n=num_classes)[y_train]
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, train_size=0.5)
y_cv = np.identity(n=num_classes)[y_cv]

#MY MODEL
features = X_train.shape[1]
net = Sequential(input_dim=X_train.shape[1], layers=[
    Sigmoid(24, name="layer1"),
    Sigmoid(12, name="layer2"),
    Sigmoid(2, name="layer3"),
    Softmax(2, name="output_layer")
])

loss = BinaryCrossEntropy(net)
optimizer = Adam(net, lr=0.01)

net.summary()
net.compile(loss, optimizer)

'''history = net.fit(X_train, y_train, epochs=100, batch_size=512, validation=True)
loss_history = history["loss"]
val_loss_history = history["val_loss"]'''

train_history = {"loss": []}
val_history = {"loss": []}

epochs = 100
batches_num = 15
batch_size = 256

early_stopping = EarlyStopping(net, min_delta=0.01, patience=5, verbose=True, restore_best_weights=True, start_from_epoch=5)
metrics = {
    "accuracy": accuracy, 
    "precision": precision_score,
    "recall": recall_score
}


for epoch in range(epochs):
    #training step
    for batch_x, batch_y in create_mini_batches(X_train, y_train, batches_num, batch_size):
        y_pred = net.forward(batch_x)
        j = loss(y_pred, batch_y)
        loss.backward()
        optimizer.update()
        y_pred_bin = y_pred.argmax(axis=1)
        batch_y_bin = batch_y.argmax(axis=1)
        '''
        acc = accuracy(y_pred_bin, batch_y_bin)
        p = precision_score(y_pred_bin, batch_y_bin)
        r = recall_score(y_pred_bin, batch_y_bin)
        '''
        
    train_history["loss"].append(j)
    #validation
    y_pred_val = net.forward(X_cv)
    j_val = loss(y_pred_val, y_cv)
    val_history["loss"].append(j_val)
    
    for metric_name, metric_func in metrics.items():
        if not metric_name in train_history:
            train_history[metric_name] = []
        train_history[metric_name].append(metric_func(y_pred_bin, batch_y_bin))
        if not metric_name in val_history:
            val_history[metric_name] = []
        val_history[metric_name].append(metric_func(y_pred_val, y_cv))

    #early_stopping
    if early_stopping(epoch):
        break
    

loss_history = train_history["loss"]
val_loss_history = val_history["loss"]
acc_history = train_history["accuracy"]
val_acc_history = val_history["accuracy"]
recall_history = train_history["recall"]
precision_history = train_history["precision"]

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
print(f"Accuracy: {acc}%")

'''
plt.plot(recall_history, precision_history, color="orange")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-recall curve")
plt.show()
'''
net.save("model_data")
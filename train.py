import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from nn.train_test_split import *
from nn.models import Sequential
from nn.layers import Sigmoid, Softmax, ReLU, LeakyReLU, Tanh
from nn.loss import BinaryCrossEntropy, CategoricalCrossEntropy
from nn.optimizers import  Adam
from nn.create_minibatches import *
from nn.metrics import *
from nn.regularizers import L2
from nn.callbacks import EarlyStopping
import seaborn as sns
import time

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
    Sigmoid(64, name="layer1"),
    Sigmoid(32, name="layer2"),
    Sigmoid(24, name="layer3"),
    Sigmoid(12, name="layer4"),
    Sigmoid(2, name="layer5"),
    Softmax(2, name="output_layer")
])

criterion = BinaryCrossEntropy()
optimizer = Adam(net, lr=0.001)

#net.summary()
net.compile(criterion, optimizer)

'''history = net.fit(X_train, y_train, epochs=100, batch_size=512, validation=True)
loss_history = history["loss"]
val_loss_history = history["val_loss"]'''

train_history = {"loss": [], "accuracy": []}
val_history = {"loss": [], "accuracy": []}

epochs = 100
batch_size = 64
m, n = X_train.shape
early_stopping = EarlyStopping(net, min_delta=0.01, patience=5, verbose=True, restore_best_weights=True, start_from_epoch=5)
parameters = net.parameters()

for epoch in range(epochs):
    #training step
    mini_batches = create_minibatches(X_train, y_train, batch_size)
    mini_batches_num = len(mini_batches)
    epoch_loss = 0
    epoch_acc = 0
    before_train_time = time.time()
    for batch_x, batch_y in mini_batches:    
        #train
        y_pred = net.forward(batch_x) #que devuelva cache con a y z de cada capa
        loss = criterion(y_pred, batch_y)
        grad_loss = criterion.grad_loss(y_pred, batch_y)
        net.backward(grad_loss) #backward como funcion del modelo y que devuelva parametros
        optimizer.update()
        y_pred_bin = y_pred.argmax(axis=1)
        batch_y_bin = batch_y.argmax(axis=1)
        epoch_loss += loss
        epoch_acc += accuracy(y_pred_bin, batch_y_bin)
    after_train_time = time.time()   
    print(f"Epoch {epoch} | {after_train_time - before_train_time:2f}s")
    epoch_loss /= mini_batches_num 
    epoch_acc /= mini_batches_num   
    train_history["loss"].append(epoch_loss)
    train_history["accuracy"].append(epoch_acc)
    #validation
    y_pred_val = net.forward(X_cv)
    val_loss = criterion(y_pred_val, y_cv)
    y_pred_val_bin = y_pred_val.argmax(axis=1)
    y_cv_bin = y_cv.argmax(axis=1)
    val_acc = accuracy(y_pred_val_bin, y_cv_bin)
    val_history["loss"].append(val_loss)
    val_history["accuracy"].append(val_acc)
    #early_stopping
    if early_stopping(epoch_loss):
        break
    

loss_history = train_history["loss"]
val_loss_history = val_history["loss"]
acc_history = train_history["accuracy"]
val_acc_history = val_history["accuracy"]

print("Train Loss after training:", loss_history[-1])
print("Validation Loss after training:", val_loss_history[-1])
print(f"Validation accuracy after training: {val_acc_history[-1]}%")

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
y_prob = predictions_onehot[:, 1]
predictions = predictions_onehot.argmax(axis=1)

acc = accuracy(predictions, y_test)
print(f"Test Accuracy: {acc}%")
f1 = f1_score(predictions, y_test)
print("f1 score:", f1)

#ROC CURVE
fpr, tpr, _ = roc_curve(y_prob, y_test)
auc_ = auc(fpr, tpr)
print("my auc:", auc_)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_:.2f})', clip_on=False)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve")
plt.show()

#PRECISION-RECALL CURVE
no_skill = len(y_test[y_test==1]) / len(y_test)
precision, recall,  _ = precision_recall_curve(y_prob, y_test)
plt.plot(precision, recall, color='darkorange', lw=2)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("Precision-Recall curve")
plt.show()

net.save("model_data")
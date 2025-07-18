import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from nn.train_test_split import *
from nn.models import *
from nn.layers import ReLU, Linear, Softmax
from nn.loss import BinaryCrossEntropy
from nn.optimizers import Adam
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
#MY MODEL
features = X_train.shape[1]

'''
net = Sequential(input_dim=X_train.shape[1], layers=[
    ReLU(64, name="layer1"),
    ReLU(32, name="layer2"),
    Linear(2, name="layer4"),
    Softmax(2, name="output_layer")
])

criterion = BinaryCrossEntropy()
optimizer = Adam(net, lr=0.05)
metrics = {"accuracy": categorical_accuracy}
'''

net = create_net_from_file("net_def.json")

early_stopping = EarlyStopping(net, min_delta=0.01, patience=10, verbose=True, restore_best_weights=True, start_from_epoch=5)
#net.compile(criterion, optimizer, metrics)

history = net.fit(X_train, y_train, epochs=100, batch_size=64, validation=True, validation_data=(X_cv, y_cv), validation_batch_size=32, verbose=1)

loss_history = history.train["loss"]
val_loss_history = history.validation["loss"]
acc_history = history.train["accuracy"]
val_acc_history = history.validation["accuracy"]

print("Train Loss after training:", loss_history[-1])
print("Validation Loss after training:", val_loss_history[-1])
print(f"Validation accuracy after training: {val_acc_history[-1]}%")

#LOSS PLOT
plt.plot(list(range(len(loss_history))), loss_history, color="orange")
plt.plot(list(range(len(val_loss_history))), val_loss_history, color="blue", linestyle="dotted", alpha=0.8)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss history")
plt.legend(labels=["training", "validation"])
plt.show()

#ACC PLOT
plt.plot(list(range(len(acc_history))), acc_history, color="orange")
plt.plot(list(range(len(val_acc_history))), val_acc_history, color="blue", linestyle="dotted", alpha=0.8)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy history")
plt.legend(labels=["training", "validation"])
plt.show()

y_train = y_train_bin
y_probs = net.predict(X_train)
y_true_prob = y_probs[:, 1]
y_pred = y_probs.argmax(axis=1)
precision = precision_score(y_pred, y_train)
recall = recall_score(y_pred, y_train)
f1 = f1_score(y_pred, y_train)
print("Precision score:", precision)
print("Recall score:", recall)
print("f1 score:", f1)

#ROC CURVE
fpr, tpr, _ = roc_curve(y_true_prob, y_train)
auc_ = auc(fpr, tpr)
print("my auc:", auc_)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_:.2f})', clip_on=False)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No Skill')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve")
plt.show()

#PRECISION-RECALL CURVE
no_skill = len(y_train[y_train == 1]) / len(y_train)
precision, recall,  _ = precision_recall_curve(y_true_prob, y_train)
plt.plot(recall, precision, color='darkorange', lw=2)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.show()

#SERIALIZE MODEL
save(net, "my_model")

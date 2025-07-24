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

#X_test = dataset.drop(columns=[0, 10, 12, 15, 19, 20]).to_numpy()
X_test = dataset[[2, 3, 4, 5, 6, 7, 8, 22, 24, 28]].to_numpy()
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
acc = accuracy(y_pred, y_test)
precision = precision_score(y_pred, y_test)
recall = recall_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test)

print(f"Loss: {loss}")
print(f"Accuracy: {acc}%")
print("Precision score:", precision)
print("Recall score:", recall)
print("f1 score:", f1)

#ROC CURVE
fpr, tpr, _ = roc_curve(y_true_prob, y_test)
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
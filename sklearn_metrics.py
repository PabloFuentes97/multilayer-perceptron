from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nn.models import Sequential
from nn.layers import Sigmoid, Softmax

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

net = Sequential(input_dim=X_train.shape[1], layers=[
    Sigmoid(64, name="layer1"),
    Sigmoid(32, name="layer2"),
    Sigmoid(24, name="layer3"),
    Sigmoid(12, name="layer4"),
    Sigmoid(2, name="layer5"),
    Softmax(2, name="output_layer")
])
net.load("model_data")
predictions_onehot = net.forward(X_test)
y_prob = predictions_onehot[:, 1]

#ROC-AUC CURVE
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = auc(fpr, tpr)
print("sklearn auc:", auc)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})', clip_on=False)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve sklearn")
plt.show()

#PRECISION-RECALL CURVE
recall, precision, _ = precision_recall_curve(y_test, y_prob)
plt.plot(precision, recall, color='darkorange', lw=2)
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("sklearn Recall-Precision curve")
plt.show()

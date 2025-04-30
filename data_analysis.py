import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr

dataset = pd.read_csv("data.csv", header=None)
features = dataset.columns
X = dataset.drop(columns=[0, 1])
y = np.array([True if diagnosis == "M" else False for diagnosis in dataset[1]])

#POINT BISERIAL CORRELATION OF ALL DATA -> BETWEEN CONTINOUS AND CATEGORICAL VARIABLES
correlated_features = []
for i in range(X.shape[1]):
    X_i = X.iloc[:, i]
    rpbis_i, p = pointbiserialr(y, X_i)
    if p < 0.05:
        correlated_features.append((i, rpbis_i))
correlated_features.sort(key=lambda tup: tup[1], reverse=True)

print("Correlated features:")
for feature, corr in correlated_features:
    print(feature, ":", corr)
    
#HISTOGRAMS
fig, ax = plt.subplots(5, 6, figsize=(15, 12))
ax = ax.flatten()
idx_malign = np.where(y == True)
idx_benign = np.where(y == False)
n, m = X.shape
for i in range(m):
    X_i = X.iloc[:, i].to_numpy()
    X_malign = X_i[idx_malign]
    X_benign = X_i[idx_benign]
    ax[i].hist(X_malign, bins=5, alpha=0.5, color="red")
    ax[i].hist(X_benign, bins=5, alpha=0.5, color="blue")
    ax[i].set_title(f"{features[i]}") 
fig.legend(labels=["malign", "benign"])
fig.tight_layout()
plt.show()

#VIOLIN PLOT
fig, ax = plt.subplots(5, 6, figsize=(15, 12))
ax = ax.flatten()
idx_malign = np.where(y == True)
idx_benign = np.where(y == False)
n, m = X.shape
quantiles = [0.25, 0.5, 0.75]
for i in range(m):
    X_i = X.iloc[:, i].to_numpy()
    X_malign = X_i[idx_malign]
    X_benign = X_i[idx_benign]
    ax[i].violinplot(X_malign, vert=False, showmeans=True, showextrema=True, quantiles=quantiles)
    ax[i].violinplot(X_benign, vert=False, showmeans=True, showextrema=True, quantiles=quantiles)
    ax[i].set_title(f"{features[i]}")
fig.legend(labels=["malign", "benign"])
fig.tight_layout()
plt.show()
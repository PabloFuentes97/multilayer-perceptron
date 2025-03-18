import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
'''
from nn.train_test_split import *
#from my_mlp import *
from nn.models import Sequential
from nn.layers import Sigmoid, Softmax
from nn.loss import CategoricalCrossEntropy
from nn.optimizers import Adam
from nn.metrics import accuracy
import seaborn as sns

dataset = pd.read_csv("data.csv", header=None)
dataset[1] = [1 if result == "M" else 0 for result in dataset[1]]
print(dataset.describe())
vars = dataset.columns
vars = vars.delete(1)

X = dataset.drop(columns=[1]).to_numpy()
y = dataset[1].to_numpy().reshape(-1, 1)
fig, ax = plt.subplots(4, 8)
ax = ax.flatten()
y_f = y.flatten()
idx_m = np.where(y_f == 1)
idx_b = np.where(y_f == 0)
n, m = X.shape
print("X shape:", X.shape)
for i in range(m):
    X_i = X[:, i]
    X_m = X_i[idx_m]
    X_b = X_i[idx_b]
    ax[i].hist(X_m, alpha=0.5, color="red")
    ax[i].hist(X_b, alpha=0.5, color="blue")
    ax[i].set_title(f"{i}")

fig.legend(labels=["malign", "benign"])    
plt.show()

'''
grid = sns.PairGrid(dataset, hue=1, vars=vars, height=2)
grid.map_diag(sns.histplot)
grid.map_offdiag(sns.scatterplot)

ax = sns.pairplot(dataset, hue=1, vars=vars)
plt.savefig("pair_plot.png")
'''
classes = np.unique(y)
num_classes = len(classes)
y_num = np.array([1 if result == "M" else 0 for result in y])
print("y:", y_num)
y_matrix = np.identity(n=num_classes)[y_num]
print("ONE HOT ENCODED Y:", y_matrix)
print("X shape:", X.shape)
scaler = StandardScaler()
Xn = scaler.fit_transform(X)
print("Xn shape:", Xn.shape)

print("CLASSES:", classes)

X_train, X_, y_train, y_ = train_test_split(Xn, y_matrix, train_size=0.6)
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, train_size=0.5)

#TENSORFLOW 
'''
print("TENSORFLOW")
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(31,)),
        Dense(units=24, activation='sigmoid', name='layer1'),
        Dense(units=24, activation='sigmoid', name='layer2'),
        Dense(units=2, activation="softmax", name='output_layer')
     ]
)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1),
    metrics=["accuracy"]
)
print(model.summary())

#y_train_cat = to_categorical(y_train, num_classes=num_classes)
history = model.fit(
    X_train, y_train,            
    epochs=100,
)
np.set_printoptions(precision=3, suppress=True)
#LOSS PLOT
plt.plot(list(range(100)), history.history["loss"], color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss history TensorFlow model")
plt.show()

#ACCURACY PLOT
plt.plot(list(range(100)), history.history["accuracy"], color="red")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy history TensorFlow model")
plt.show()

proba = model.predict(X_test)
#print("PROBA:", proba)
predictions = np.argmax(proba, axis=1)
y_test_pred = np.argmax(y_test, axis=1)
#print("PREDICTIONS:", predictions)
acc = (y_test_pred == predictions).mean()
print(f"Accuracy: {acc * 100}%")
'''

#MY MODEL
input_features = X_train.shape[1]
net = Sequential([
    Sigmoid(31, 8, name="layer1"),
    Sigmoid(8, 2, name="layer2"),
    Softmax(2, 2, name="output_layer")
])


loss = CategoricalCrossEntropy(net)
optimizer = Adam(net, lr=0.1)

net.summary()
net.compile(loss, optimizer)
'''
loss_history = []
batches = 15
batch_size = 32
epochs = 100
m, n = X_train.shape

# Early stopping parameters
patience = 5 # Number of epochs to wait before stopping
min_delta = 0.01 # Minimum change in loss to qualify as an improvement
patience_counter = 0
best_loss = float('inf')

for epoch in range(epochs):
    #for _ in range(batches):
    batch_idx = np.random.randint(m, size=batch_size)
    X_train_batch = X_train[batch_idx]
    y_train_batch = y_train[batch_idx]
    y_pred = net.forward(X_train_batch)
    j = loss(y_pred, y_train_batch)
    loss.backward()
    optimizer.update(epoch)
    
    # Check for improvement
    if j < best_loss - min_delta:
        best_loss = j
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch} |", "Loss:", j)
    loss_history.append(j)
'''

history = net.fit(X_train, y_train, 100)
loss_history = history["loss"]
print("Loss after training:", loss_history[-1])

plt.plot(list(range(len(loss_history))), loss_history, color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss history")
plt.show()


predictions_onehot = net.predict(X_test)
#print("Predictions one-hot-coded:", predictions_onehot)
predictions = predictions_onehot.argmax(axis=1)
y_test_pred = y_test.argmax(axis=1)
#print("Predictions:", predictions)
acc = accuracy(y_test_pred, predictions)
print(f"Accuracy: {acc * 100}%")
'''
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()'''
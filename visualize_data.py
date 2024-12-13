import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from train_test_split import *
from my_mlp import *

dataset = pd.read_csv("data.csv", header=None)

X = dataset.drop(columns=[1]).to_numpy()
y = dataset[1].to_numpy().reshape(-1, 1)

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

X_train, X_test, y_train, y_test = train_test_split(Xn, y_matrix)

#TENSORFLOW 
'''
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)
print(model.summary())

y_train_cat = to_categorical(y_train, num_classes=num_classes)
model.fit(
    X_train, y_train_cat,            
    epochs=100,
)
np.set_printoptions(precision=3, suppress=True)
proba = model.predict(X_test)
print("PROBA:", proba)
predictions = np.argmax(proba, axis=1)
print("PREDICTIONS:", predictions)
acc = (y_test == predictions).mean()
print(f"Accuracy: {acc * 100}%")
'''

#MY MODEL
input_features = X_train.shape[1]
net = MultilayerPerceptron([
    Sigmoid(31, 24),
    Sigmoid(24, 24),
    Softmax(24, 2)
])
    
optimizer = SGD(net, lr=0.1)
loss = CategoricalCrossEntropy(net)

net.summary()
    
epochs = 1000
for epoch in range(epochs):
    y_pred = net.forward(X_train)
    j = loss(y_pred, y_train)
    loss.backward()
    optimizer.update()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} |", "Loss:", j)

predictions = net.forward(X_test).argmax(axis=1)
print("Predictions:", predictions)
acc = (y_test == predictions).mean()
print(f"Accuracy: {acc * 100}%")
'''
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()'''
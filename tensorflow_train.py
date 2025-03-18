import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

tf.random.set_seed(1234)  # applied to achieve consistent results

#PROCESS DATASET
dataset = pd.read_csv("data.csv", header=None)
dataset[1] = [1 if result == "M" else 0 for result in dataset[1]]
print(dataset.describe())
features = dataset.columns

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
#print("PREDICTIONS:", predictions)
acc = (y_test == predictions).mean()
print(f"Accuracy: {acc * 100}%")
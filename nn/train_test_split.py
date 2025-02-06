import numpy as np
import pandas as pd
import typing
import copy

def train_test_split(X: np.ndarray, y: np.ndarray, train_size=0.8):
    if train_size >= 1:
        raise Exception
    
    size = len(X)
    batch_size = int(size * train_size)
    batch = np.random.randint(0, size, batch_size) 
    train_X = X[batch]
    test_X = X[batch]
    train_y = y[batch]
    test_y = y[batch]
    
    return train_X, test_X, train_y, test_y

if __name__ == "__main__":
    dataset = pd.read_csv("data.csv", header=None)
    X = dataset.drop(columns=[1]).to_numpy()
    y = dataset[1].to_numpy()
    print("X:", X)
    print("y:", y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    print("X_train:", X_train)
    print("X_test:", X_test)
    print("y_train:", y_train)
    print("y_test:", y_test)
    
    print("DATASET SIZE:", len(dataset), ", TRAIN + TEST:", len(X_train) + len(y_test))

import numpy as np
from .train_test_split import train_test_split
from termcolor import colored
import time
class Sequential:
    def __init__(self, layers):
        #red como lista de capas
        self.layers = layers
        self.stop_training = False

        '''
        for idx, layer in self.layers:
            if idx > 0:
        '''        
        
    def compile(self, loss, optimizer):
        self.loss = loss
        self.loss.net = self
        self.optimizer = optimizer
        self.optimizer.net = self
    
    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.get_weights())
            
        return weights
    
    def set_weights(self, weights):
        for layer, w  in zip(self.layers, weights):
            layer.set_weights(w)
        
    def fit(self, X, y, epochs, batches=15, batch_size=32, validation=False):
        
        loss_history = []
        acc_history = []
        m = X.shape[0]
        
        if validation:
            X, X_val, y, y_val = train_test_split(X, y)
            m = X.shape[0]
            val_loss_history = []
            val_acc_history = []
            val_m = X_val.shape[0]
            val_batch_size = val_m if val_m < batch_size else batch_size
            
        for epoch in range(epochs):
            before_epoch_time = time.time()
            for mini_batch in range(batches):
                batch_idx = np.random.randint(m, size=batch_size)
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                y_pred = self.forward(X_batch)
                j = self.loss(y_pred, y_batch)
                self.loss.backward()
                self.optimizer.update(epoch)
                acc = (y_pred.argmax(axis=1) == y_batch.argmax(axis=1)).mean()
                
                if validation:
                    batch_idx = np.random.randint(val_m, size=val_batch_size)
                    X_val_batch = X_val[batch_idx]
                    y_val_batch = y_val[batch_idx]
                    y_val_pred = self.forward(X_val_batch)
                    j_val = self.loss(y_val_pred, y_val_batch)
                    acc_val = (y_val_pred.argmax(axis=1)  == y_val_batch.argmax(axis=1)).mean()
                
                
                '''progress_bar = "|"
                print(f"\rEpoch {epoch} | Batch {mini_batch} {colored(progress_bar * mini_batch, 'green')} | Loss: {j}", end="")
                time.sleep(0.01)
            print(f"\r")'''
                '''
                # Check for improvement
                if j < best_loss - min_delta:
                    best_loss = j
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                '''
            after_epoch_time = time.time()
            epoch_time = after_epoch_time - before_epoch_time
            if epoch % 10 == 0:
                print(f"\rEpoch {epoch} | Loss: {j} | Time: {epoch_time}s")
            
            loss_history.append(j)
            acc_history.append(acc)
            
            if validation:
                val_loss_history.append(j_val)
                val_acc_history.append(acc_val)
            
        history = {"loss": loss_history,
                   "acc": acc_history
                   }
        
        if validation:
            history["val_loss"] = val_loss_history
            history["val_acc"] = val_acc_history
        
        return history
    
    def forward(self, X):
        #calcular salida del modelo aplicando cada capa secuencialmente
        for layer in self.layers:
            X = layer.forward(X)
        
        return X
        
    def summary(self):
        for idx, layer in enumerate(self.layers):
            print(f"Layer {idx}: Weights: {layer.weights.shape}, bias: 1, Output shape: (None, {layer.weights.shape[1]}), name: {layer.name}")

    def predict(self, X):
        return self.forward(X)
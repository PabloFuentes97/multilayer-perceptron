import numpy as np

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
    
    def fit(self, X, y, epochs, batch_size=32):
        
        loss_history = []
        m, n = X.shape
        
        for epoch in range(epochs):
        #for _ in range(batches):
            batch_idx = np.random.randint(m, size=batch_size)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            y_pred = self.forward(X_batch)
            j = self.loss(y_pred, y_batch)
            self.loss.backward()
            self.optimizer.update(epoch)
            
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
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch} |", "Loss:", j)
            '''
            loss_history.append(j)
            
        return {
            "loss": loss_history
                }
    
    def forward(self, X):
        #calcular salida del modelo aplicando cada capa secuencialmente
        for layer in self.layers:
            X = layer.forward(X)
        
        return X
        
    def predict(self, X):
        pass
    
    def summary(self):
        for idx, layer in enumerate(self.layers):
            print(f"Layer {idx}: Weights: {layer.weights.shape}, bias: 1, Output shape: (None, {layer.weights.shape[1]}), name: {layer.name}")

    def predict(self, X):
        return self.forward(X)
import numpy as np

class EarlyStopping:
    def __init__(self, net, min_delta=0, patience=0, verbose=0, restore_best_weights=False, start_from_epoch=0):
        self.net = net
        self.min_delta = min_delta
        self.patience = patience
        self.patience_counter = 0
        self.verbose = verbose
        #self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        if self.restore_best_weights:
            self.best_weights = self.net.get_weights()
        self.start_from_epoch = start_from_epoch
        self.epoch = 0
        self.best_loss = np.inf
        return

    def __call__(self, j):
        self.epoch += 1
        if self.epoch < self.start_from_epoch:
            return False

        # Check for improvement
        if j < self.best_loss - self.min_delta:
            self.best_loss = j
            self.patience_counter = 0
            if self.restore_best_weights:
                self.best_weights = self.net.get_weights()
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {self.epoch}")
                if self.restore_best_weights:
                    self.net.set_weights(self.best_weights)
                    return True
        
        

        
    
        
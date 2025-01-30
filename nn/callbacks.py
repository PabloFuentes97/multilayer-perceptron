import numpy



class EarlyStopping:
    def __init__(net, monitor="val_loss", min_delta=0, patience=0, verbose=0, mode="auto", baseline=None, restore_best_weights=False, start_from_epoch=0):
        self.net = net
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.patience_counter = 0
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.epoch = 0
        self.best_loss = np.inf
        return

    def start_train(self):
        if self.restore_best_weights:
            self.best_weights = self.net.get_weights()

    def __call__(self):
        self.epoch += 1
        if self.epoch < self.start_from_epoch
            return

        # Check for improvement
        if j < best_loss - self.min_delta:
            self.best_loss = j
            self.patience_counter = 0
            self.best_weights = self.net.get_weights()
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                if self.restore_best_weights:
                    self.net.set_weights(self.best_weights)
                    self.net.stop_training = True
        
        

        
    
        
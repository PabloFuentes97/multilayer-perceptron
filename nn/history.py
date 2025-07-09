class History:
    def __init__(self, metrics):
        self.metrics = metrics
        
    def on_train_begin(self):
        self.train = {"loss": []}
        self.validation = {"loss": []}
        self.train_epoch = {"loss": 0}
        self.validation_epoch = {"loss": 0}
        
        if not self.metrics:
            return 
    
        for metric in self.metrics:
            self.train[metric] = []
            self.validation[metric] = []
            self.train_epoch[metric] = 0
            self.validation_epoch[metric] = 0

    def on_train_end(self):
        delattr(self, "train_epoch")
        delattr(self, "validation_epoch")   
     
    def on_batch_begin(self):
        pass
    
    def on_batch_end(self, y_pred, y_true, loss, type):
        if type == "train":
            self.train_epoch["loss"] += loss
            if not self.metrics:
                return 
            for metric_name, metric_func in self.metrics.items():
                self.train_epoch[metric_name] += metric_func(y_pred, y_true)
                
        elif type == "validation":
            self.validation_epoch["loss"] += loss
            if not self.metrics:
                return 
            for metric_name, metric_func in self.metrics.items():
                self.validation_epoch[metric_name] += metric_func(y_pred, y_true)
            
    def on_train_epoch_begin(self):
        self.train_epoch["loss"] = 0
        if not self.metrics:
            return
        for metric in self.metrics:
            self.train_epoch[metric] = 0
    
    #cambiar esto para que itere directamente por train que tiene todas las métricas, incluido loss
    def on_train_epoch_end(self, minibatches_num):
        self.train_epoch["loss"] /= minibatches_num
        self.train["loss"].append(self.train_epoch["loss"])
        if not self.metrics:
            return 
        for metric in self.metrics:
            self.train_epoch[metric] /= minibatches_num 
            self.train[metric].append(self.train_epoch[metric])
            
    def on_validation_epoch_begin(self):
        self.validation_epoch["loss"] = 0
        if not self.metrics:
            return
        for metric in self.metrics:
            self.validation_epoch[metric] = 0
    
    def on_validation_epoch_end(self, minibatches_num):
        self.validation_epoch["loss"] /= minibatches_num
        self.validation["loss"].append(self.validation_epoch["loss"])
        if not self.metrics:
            return 
        for metric in self.metrics:
            self.validation_epoch[metric] /= minibatches_num
            self.validation[metric].append(self.validation_epoch[metric])
    
    
    
    
    
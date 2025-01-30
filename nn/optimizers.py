import numpy as np

class SGD:
    def __init__(self, net, lr=0.01, epochs=1000, batch_size=32, tolerance=0.0001, lambda_=None):
        self.net = net
        self.lr = lr
        self.epochs = epochs
        self.tolerance = tolerance
        
    def update(self):
        
        for layer in self.net.layers:
            layer.update(
                self.lr * layer.dw,  
                self.lr * layer.db
            )
            
        '''for layer in self.net.layers:
            layer.update([
                params - self.lr * grads
                for params, grads in zip(layer.params, layer.grads)
            ])'''
            
class SGDMomentum:
    def __init__(self, net, lr=0.01, epochs=1000, batch_size=32, tolerance=0.0001, lambda_=None, beta=0.9):
        self.net = net
        self.lr = lr
        self.epochs = epochs
        self.tolerance = tolerance
        self.beta = beta
        
    def update(self):
        for layer in self.net.layers:
            if not hasattr(layer, "VdW"):
                layer.VdW = 0
            if not hasattr(layer, "VdB"):
                layer.VdB = 0
            layer.VdW = self.beta * layer.VdW + (1 - self.beta) * layer.dw
            layer.VdB = self.beta * layer.VdB + (1 - self.beta) * layer.db
            layer.update(
                self.lr * layer.VdW,  
                self.lr * layer.VdB
            )
            
class RMSProp:
    def __init__(self, net, lr=0.01, epochs=1000, batch_size=32, tolerance=0.0001, lambda_=None, beta=0.9):
        self.net = net
        self.lr = lr
        self.epochs = epochs
        self.tolerance = tolerance
        self.beta = beta
        
    def update(self):
        epsilon = 1e-5
        for layer in self.net.layers:
            if not hasattr(layer, "SdW"):
                layer.SdW = 0
            if not hasattr(layer, "SdB"):
                layer.SdB = 0
            layer.SdW = self.beta * layer.SdW + (1 - self.beta) * (layer.dw ** 2)
            layer.SdB = self.beta * layer.SdB + (1 - self.beta) * (layer.db ** 2)
            layer.update(
                self.lr * (layer.dw / np.sqrt(layer.SdW + epsilon)),  #si sdW es muy grande, al dividir por él el resultado es menor -> La actualición del parámetro es más pequeña. Y viceversa
                self.lr * (layer.db / np.sqrt(layer.SdB + epsilon))
            )

class Adam:
    def __init__(self, net, lr=0.01, epochs=1000, batch_size=32, beta1=0.9, beta2=0.999):
        self.net = net
        self.lr = lr
        self.epochs = epochs
        self.beta1 = beta1
        self.beta2 = beta2
        
    def update(self, iter=1):
        epsilon = 1e-5
        for layer in self.net.layers:
            if not hasattr(layer, "VdW"):
                layer.VdW = 0
            if not hasattr(layer, "VdB"):
                layer.VdB = 0
            if not hasattr(layer, "SdW"):
                layer.SdW = 0
            if not hasattr(layer, "SdB"):
                layer.SdB = 0
                
            #Momentum
            layer.VdW = self.beta1 * layer.VdW + (1 - self.beta1) * layer.dw #en primera interación, valor del primer término es 0. 
                                                                                #En primeras iteraciones, el valor general es muy pequeño, tarda en "arrancar"
            layer.VdB = self.beta1 * layer.VdB + (1 - self.beta1) * layer.db
            #RMSProp
            layer.SdW = self.beta2 * layer.SdW + (1 - self.beta2) * (layer.dw ** 2)
            layer.SdB = self.beta2 * layer.SdB + (1 - self.beta2) * (layer.db ** 2)
            
            # correccion para compensar arranque lento
            layer.VdW = layer.VdW / ((1 - self.beta1) ** iter + 1)
            layer.VdB = layer.VdB / ((1 - self.beta1) ** iter + 1)
            layer.SdW = layer.SdW / ((1 - self.beta2) ** iter + 1)
            layer.SdB = layer.SdB / ((1 - self.beta2) ** iter + 1)
            
            layer.update(
                self.lr * (layer.VdW / np.sqrt(layer.SdW + epsilon)), #Momentum / RMSProp
                self.lr * (layer.VdB / np.sqrt(layer.SdB + epsilon))
            )
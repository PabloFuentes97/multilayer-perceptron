import numpy as np 

class SGD:
    def __init__(self, net, lr=0.01, batch_size=32, tolerance=0.0001, lambda_=None):
        self.net = net
        self.lr = lr
        self.tolerance = tolerance
        
    def update(self):
        for layer in self.net.layers:
            layer.update(
                self.lr * layer.dw,  
                self.lr * layer.db
            )
            
class SGDMomentum:
    def __init__(self, net, lr=0.01, tolerance=0.0001, lambda_=None, beta=0.9):
        self.net = net
        self.lr = lr
        self.tolerance = tolerance
        self.beta = beta
    
    def init_velocity(self, params):
        self.vdW = []
        self.vdb = []
        
        layers = params // 2
        for l in range(0, layers):
            w_i, b_i = params[l]
            vdW_i = np.zeros(shape=w_i.shape)
            vdb_i = np.zeros(shape=b_i.shape)
            self.vdW.append(vdW_i)
            self.vdb.append(vdb_i)
    

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
            
class RMSProp: #Root Mean Squared Prop
    def __init__(self, net, lr=0.01, tolerance=0.0001, lambda_=None, beta=0.9):
        self.net = net
        self.lr = lr
        self.tolerance = tolerance
        self.beta = beta
        
    def update(self):
        epsilon = 1e-5
        for layer in self.net.layers:
            if not hasattr(layer, "SdW"):
                layer.SdW = 0
            if not hasattr(layer, "SdB"):
                layer.SdB = 0
            layer.SdW = self.beta * layer.SdW + (1 - self.beta) * (layer.dw ** 2) #horizontal axis -> try to update faster
            layer.SdB = self.beta * layer.SdB + (1 - self.beta) * (layer.db ** 2) #vertical axis -> try to update slower -> dump oscilations
            layer.update(
                self.lr * (layer.dw / np.sqrt(layer.SdW + epsilon)),  #si sdW es muy grande, al dividir por él el resultado es menor -> La actualización del parámetro es más pequeña. Y viceversa
                self.lr * (layer.db / np.sqrt(layer.SdB + epsilon))
            )

class Adam: #Adaptive Moment Estimation
    def __init__(self, net, lr=0.01, beta1=0.9, beta2=0.999, decay=None, decay_rate=1):
        self.net = net
        self.parameters = net
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
       
    def update(self):
        epsilon = 1e-8
        for layer in self.net.layers:
            if not hasattr(layer, "VdW"):
                layer.VdW = np.zeros(shape=layer.weights.shape)
            if not hasattr(layer, "VdB"):
                layer.VdB = np.zeros(1)
            if not hasattr(layer, "SdW"):
                layer.SdW = np.zeros(shape=layer.weights.shape)
            if not hasattr(layer, "SdB"):
                layer.SdB = np.zeros(1)
                
            #Momentum
            layer.VdW = self.beta1 * layer.VdW + (1 - self.beta1) * layer.dw #en primera interación, valor del primer término es 0. 
                                                                                #En primeras iteraciones, el valor general es muy pequeño, tarda en "arrancar"
            layer.VdB = self.beta1 * layer.VdB + (1 - self.beta1) * layer.db
            #RMSProp
            layer.SdW = self.beta2 * layer.SdW + (1 - self.beta2) * (layer.dw ** 2)
            layer.SdB = self.beta2 * layer.SdB + (1 - self.beta2) * (layer.db ** 2)
            
            #Bias correction -> compensar arranque lento
            VdW_hat = layer.VdW / (1.0 - self.beta1 ** (self.t + 1))
            VdB_hat = layer.VdB / (1.0 - self.beta1 ** (self.t + 1))
            SdW_hat = layer.SdW / (1.0 - self.beta2 ** (self.t + 1))
            SdB_hat = layer.SdB / (1.0 - self.beta2 ** (self.t + 1))
            
            w_new = VdW_hat / (np.sqrt(SdW_hat) + epsilon) #Momentum / RMSProp
            b_new = VdB_hat / (np.sqrt(SdB_hat) + epsilon)
            
            layer.update(
                self.lr * w_new, 
                self.lr * b_new
            )
            
        self.t += 1
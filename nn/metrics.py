import numpy as np

def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean() * 100

def find_TP(y_pred, y_true):
   # counts the number of true positives (y = 1, y_hat = 1)
   return sum((y_true == 1) & (y_pred == 1))
def find_FN(y_pred, y_true):
   # counts the number of false negatives (y = 1, y_hat = 0) Type-II error
   return sum((y_true == 1) & (y_pred == 0))
def find_FP(y_pred, y_true): 
   # counts the number of false positives (y = 0, y_hat = 1) Type-I error
   return sum((y_true == 0) & (y_pred == 1))
def find_TN(y_pred, y_true):
   # counts the number of true negatives (y = 0, y_hat = 0)
   return sum((y_true == 0) & (y_pred == 0))

def create_confussion_matrix(y_pred, y_true):
    cm = {}
    cm["tp"] = find_TP(y_pred, y_true)
    cm["fn"] = find_FN(y_pred, y_true)
    cm["fp"] = find_FP(y_pred, y_true)
    cm["tn"] = find_TN(y_pred, y_true)
    
    return cm

def precision_score(y_pred, y_true):
    tp = find_TP(y_pred, y_true)
    fp = find_FP(y_pred, y_true)
    return tp / (tp + fp)

def recall_score(y_pred, y_true):
    tp = find_TP(y_pred, y_true)
    fn = find_FN(y_pred, y_true)
    
    return tp / (tp + fn)
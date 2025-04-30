import numpy as np
import matplotlib.pyplot as plt
import copy

def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean() * 100

def find_TP(y_pred, y_true):
   # counts the number of true positives (y = 1, y_hat = 1)
   return np.sum((y_true == 1) & (y_pred == 1))
def find_FN(y_pred, y_true):
   # counts the number of false negatives (y = 1, y_hat = 0) Type-II error
   return np.sum((y_true == 1) & (y_pred == 0))
def find_FP(y_pred, y_true): 
   # counts the number of false positives (y = 0, y_hat = 1) Type-I error
   return np.sum((y_true == 0) & (y_pred == 1))
def find_TN(y_pred, y_true):
   # counts the number of true negatives (y = 0, y_hat = 0)
   return np.sum((y_true == 0) & (y_pred == 0))

def find_FPR(y_pred, y_true):
   fp = find_FP(y_pred, y_true)
   tn = find_TN(y_pred, y_true)
   
   fpr = fp / (fp + tn)
   
   return fpr

def find_TPR(y_pred, y_true):
   tp = find_TP(y_pred, y_true)
   fn = find_FN(y_pred, y_true)
   
   tpr = tp / (tp + fn)
   
   return tpr

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
   
   if (tp + fp) == 0: #no positives predicted
      return 0.0
   
   return tp / (tp + fp) # true positives / total estimated positives

def recall_score(y_pred, y_true):
   tp = find_TP(y_pred, y_true)
   fn = find_FN(y_pred, y_true)
    
   if (tp + fn) == 0: #no positives detected
      return 0.0
   
   return tp / (tp + fn) # true positives / total examples

def f1_score(y_pred, y_true):
   precision = precision_score(y_pred, y_true)
   recall = recall_score(y_pred, y_true)
   
   return 2 * (precision * recall / (precision + recall))

def roc_curve(y_pred, y_true):
   fpr = []
   tpr = []
   thresholds = np.concatenate(([np.inf],np.sort(y_pred)[::-1]))

   for t in thresholds:
      y_threshold = y_pred >= t
      fpr_t = find_FPR(y_threshold, y_true)
      tpr_t = find_TPR(y_threshold, y_true)
      fpr.append(fpr_t)
      tpr.append(tpr_t)
   
   return fpr, tpr, thresholds
      
def auc(fpr, tpr):
   return np.trapezoid(tpr, fpr)
      
def roc_auc_score(y_pred, y_test):
   fpr, tpr, _ = roc_curve(y_pred, y_test)
   return auc(fpr, tpr)

def precision_recall_curve(y_pred, y_true):
   recall = []
   precision = []
   thresholds = np.concatenate(([np.inf], np.sort(y_pred)[::-1]))

   for t in thresholds:
      y_threshold = y_pred >= t
      r = recall_score(y_threshold, y_true)
      p = precision_score(y_threshold, y_true)
      recall.append(r)
      precision.append(p)
      
   return recall, precision, thresholds
import numpy as np

def create_minibatches(X, y, batch_size):
    mini_batches = []
    m = X.shape[0]
    if batch_size > m:
        batch_size = m
    
    complete_batches_num = m // batch_size
    shuffle_idx = list(np.random.permutation(m))
    shuffle_X = X[shuffle_idx]
    shuffle_y = y[shuffle_idx]

    for k in range(0, complete_batches_num):
        i = k * batch_size
        j = (k + 1) * batch_size
    
        mini_batch_X = shuffle_X[i: j,:]
        mini_batch_Y = shuffle_y[i: j,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % batch_size != 0:
        i = complete_batches_num * batch_size
        j = i + m % batch_size 
        
        mini_batch_X = shuffle_X[i: j,:]
        mini_batch_Y = shuffle_y[i: j,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
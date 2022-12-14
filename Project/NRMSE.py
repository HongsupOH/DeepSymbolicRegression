import numpy as np
from evaluateAgraph import evaluateAgraph

def reward(root,idx_to_op,X,y,constants,optimizer=False):
    reward = 1/(1+NRMSE(root,idx_to_op,X,y,constants,optimizer=optimizer))
    if optimizer==True:
        reward*=-1
    return reward

def NRMSE(root,idx_to_op,X,y,constants,optimizer=False):
    sig_y = np.std(y)
    y_hat = evaluateAgraph(root,idx_to_op,X,constants,optimizer=optimizer)
    
    RMSE = np.sqrt(np.mean((y_hat - y)**2))
    NRMSE = (1/sig_y)*RMSE
    return NRMSE

def RMSE(root,idx_to_op,X,y,constants):
    y_hat = evaluateAgraph(root,idx_to_op,X,constants)
    RMSE = np.sqrt(np.mean((y_hat - y)**2))
    return RMSE


    




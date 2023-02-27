import numpy as np
from scipy.optimize import minimize
from NRMSE import NRMSE,reward


def localOptimizer(agraph,root,X,y,idx_to_op):
    constants = np.ones(agraph.constant_count)
    objfn = lambda constants: reward(root,idx_to_op,X,y,constants,optimizer=True)
    
    result = minimize(objfn,constants,\
                   method="L-BFGS-B",\
                   options={'gtol': 1e-10})
        
    constants = result.x
    return constants

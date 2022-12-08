import numpy as np
from scipy.optimize import minimize
from evalAgraph import evaluateAgraph

def local_optimizer(OpTree,num_c,result,x_0,x_1):
    root = OpTree.root
    constants = np.ones(num_c)
    mode = 'local_opt'
    
    objfn = lambda x: np.sqrt(np.mean((evaluateAgraph(root,mode,constants=x, x_0=x_0, x_1=x_1) - result)**2))
    
    #bnds = [(0,1) for x  in range(num_c)]
    res = minimize(objfn,constants,\
                   method="L-BFGS-B",callback=new_callback(),\
                   options={'gtol': 1e-10})
        
    constants = res.x
    loss = objfn(constants)
    return constants,loss

def new_callback():
    step = 1

    def callback(xk):
        nonlocal step
        #print('Step #{}: xk = {}'.format(step, xk))
        step += 1

    return callback
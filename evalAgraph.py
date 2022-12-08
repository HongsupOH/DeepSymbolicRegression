import numpy as np
from helper import _is,sampling

def evaluateAgraph(root, mode, constants=None,x_0=None,x_1=None):
    """
    ref: https://www.geeksforgeeks.org/evaluation-of-expression-tree/
    """
    constants = constants
    x_0,x_1 = x_0,x_1
    
    if root is None:
        return 0
    
    if _is(root.op) =='unary':
        child_sum = evaluateAgraph(root.child,mode,constants,x_0,x_1)
    
    else:
        if root.left is None and root.right is None:
            if root.op == 9:
                if mode=='numeric':
                    ans = np.ones(len(x_0))*np.random.rand()
                elif mode=='local_opt':
                    ind = root.constant-1
                    ans = np.ones(len(x_0))*constants[ind]
                elif mode=='symbolic':
                    ans = 'c_{}'.format(root.constant-1)
                return ans
            elif root.op ==10:
                if mode=='symbolic':
                    return 'x_0'
                else:    
                    return x_0
            elif root.op ==11:
                if mode=='symbolic':
                    return 'x_1'
                else:    
                    return x_1
        else:    
            left_sum = evaluateAgraph(root.left,mode,constants,x_0,x_1)
            right_sum = evaluateAgraph(root.right,mode,constants,x_0,x_1)

    
    if root.op == 3:
        if mode=='symbolic':
            return '('+left_sum + '+' +right_sum +')'
        else:
            return left_sum + right_sum
    elif root.op == 4:
        if mode=='symbolic':
            return '('+left_sum + '-' + right_sum+')'
        else:
            return left_sum - right_sum
 
    elif root.op == 5:
        if mode =='symbolic':
            return '('+left_sum + '*' + right_sum+')'
        else:
            return left_sum * right_sum
 
    elif root.op == 6:
        if mode=='symbolic':
            return '('+left_sum + '/' + right_sum +')'
        else:
            return left_sum / right_sum
    
    elif root.op == 7:
        if mode=='symbolic':
            return 'sin({})'.format(child_sum)
        else:
            return np.sin(child_sum)
    
    elif root.op == 8:
        if mode=='symbolic':
            return 'cos({})'.format(child_sum)
        else:
            return np.cos(child_sum)
    
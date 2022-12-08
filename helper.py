import numpy as np
from Agraph import binary_node, unary_node

def _is(op):
    if op ==9:
        return 'constant'
    elif op>=10:
        return 'variable'
    elif op==7 or op==8:
        return 'unary'
    else:
        return 'binary'

def sampling(span = 1):
    if span == 1:
        """
        1. All nodes
        """
        op = np.random.choice([3,4,5,6,7,8,9,10,11],1)[0]
        if _is(op)=='unary':
            childNode = unary_node(op)
        else:
            childNode = binary_node(op)
    elif span == 2:
        """
        2. Binary nodes
        """
        op = np.random.choice([3,4,5,6,9,10,11],1)[0]
        childNode = binary_node(op)
    elif span == 3:
        """
        3. Unary + (*) operator
        """
        op = np.random.choice([5,7,8,9,10,11],1)[0]
        if _is(op)=='unary':
            childNode = unary_node(op)
        else:
            childNode = binary_node(op)
    elif span == 4:
        """
        4. Leaf (constant & variable)
        """
        op = np.random.choice([9,10,11],1)[0]
        childNode = binary_node(op)
        
    return op,childNode
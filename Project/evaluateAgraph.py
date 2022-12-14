import numpy as np

def evaluateAgraph(root,idx_to_op,X,constants,optimizer=False):
    
    def recursive(root,optimizer):
        
        if root is None:
            return 0
        
        if root.left is None and root.right is None:
            
            if idx_to_op[root.op] =='c':
                
                if optimizer==True:
                    ind = root.constant_index
                    return constants[ind]*np.ones(len(X))
                else:
                    return root.constant_data*np.ones(len(X))
                
            elif idx_to_op[root.op] =='x':
                return X
           
        else:
            
            left_sum = recursive(root.left,optimizer)
            right_sum = recursive(root.right,optimizer)
            
        if idx_to_op[root.op] == '+':
            return left_sum + right_sum
    
        elif idx_to_op[root.op] == '-':
            return left_sum - right_sum
    
        elif idx_to_op[root.op] == '*':
            return left_sum * right_sum
    
        elif idx_to_op[root.op] == '/':
            return left_sum / right_sum
    
    answer = recursive(root,optimizer)
    return answer
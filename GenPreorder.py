from helper import _is,sampling

def GenPreorder(OpTree):
    preorder_array = [1]
    root = OpTree.root
    
    def preorder(root):

        if root:
            
            #print(root.op, data_dict['idx_to_op'][root.op])
            preorder_array.append(root.op)
            
            if _is(root.op)=='unary':
                preorder(root.child)
            else:
                preorder(root.left)
                preorder(root.right)
                
    preorder(root)
    preorder_array.append(2)
    return preorder_array 

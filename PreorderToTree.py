import numpy as np
from Agraph import binary_node,unary_node,Tree
from helper import _is,sampling

def preorder_to_tree(preorder):
    num_zero = np.count_nonzero(preorder==0)
    preorder = preorder[1:len(preorder) - (num_zero + 1)]
    
    op = preorder[0]
    Node = binary_node(op)
    OpTree = Tree()
    OpTree.root = Node
    con_cnt = 1
    
    if op>=9:
        return OpTree
    
    stack = [Node]
    for i in range(1,len(preorder)):
        
        op = preorder[i]
        Node = binary_node(op)
        prev_node = stack[-1]
        if _is(prev_node.op)=='binary':
            if prev_node.left==None:
                prev_node.left = Node
            else:
                prev_node.right = Node
                stack.pop()
        elif _is(prev_node.op)=='unary':
            prev_node.child = Node
            stack.pop()
        
        if op<9:
            stack.append(Node)
        else:
            if _is(op)=='constant':
                Node.constant = con_cnt
                con_cnt+=1
    return OpTree

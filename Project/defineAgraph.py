def defineAgraph(agraph,constants,idx_to_op):
    root = agraph.root
    
    def preorder(root,constants,idx_to_op):

        if root:
            if idx_to_op[root.op]=='c':
                root.constant_data = constants[root.constant_index]
            
            preorder(root.left,constants,idx_to_op)
            preorder(root.right,constants,idx_to_op)
                
    preorder(root,constants,idx_to_op)
    return 
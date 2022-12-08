from helper import _is,sampling
from Agraph import binary_node,unary_node,Tree


def GenOperator(OpTree,complexity,root,nodes,span):
    
    #print('Node:{}'.format(root.op))
    if len(nodes) >= complexity:
        return
    
    if _is(root.op)== 'constant' or _is(root.op)== 'variable':
        return
    elif _is(root.op)=='unary':
        op,ChildNode = sampling(span=span)
        nodes.append(op)
        root.child = ChildNode
        GenOperator(OpTree,complexity,ChildNode,nodes,span)
    else: 
        op1,LeftNode = sampling(span=span)
        nodes.append(op1)
        root.left = LeftNode

        op2,RightNode = sampling(span=span)
        nodes.append(op2)
        root.right = RightNode

        GenOperator(OpTree,complexity,LeftNode,nodes,span)
        GenOperator(OpTree,complexity,RightNode,nodes,span)

def organizeTree(OpTree):
    root = OpTree.root
    queue = [root]
    con_cnt = 1
    
    while len(queue)>0:
        
        cur_node = queue.pop(0)
        #print(cur_node.op,data_dict['idx_to_op'][cur_node.op])
        
        if _is(cur_node.op)=='unary':
            child = cur_node.child
            if child == None:
                op,ChildNode = sampling(span = 4)
                cur_node.child = ChildNode
                child = ChildNode
            
            queue.append(child)
                
        else:
            left = cur_node.left
            right = cur_node.right
        
            if left!=None and right!=None:
                queue.append(left)
                queue.append(right)
            
            else: 
                if _is(cur_node.op)!='constant' and _is(cur_node.op)!='variable':
                    op,_ = sampling(span = 4)
                    cur_node.op = op
            
                if _is(cur_node.op) == 'constant':
                    cur_node.constant = con_cnt
                    con_cnt+=1
        
        
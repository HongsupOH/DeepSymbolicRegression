import math
import torch
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter

from Agraph import Agraph,binary_node
from localOptimizer import localOptimizer
from defineAgraph import defineAgraph
from NRMSE import reward
#from stringAgraph import stringAgraph

def rnn_step_forward(x, prev_h, Wx, b, Wh):
    fW = torch.mm(x,Wx) + torch.mm(prev_h,Wh) + b
    next_h = torch.tanh(fW)
    return next_h

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    _,H = prev_h.shape
    h = torch.mm(x,Wx) + torch.mm(prev_h,Wh) + b
    
    it = torch.sigmoid(h[:,:H])
    ft = torch.sigmoid(h[:,H:2*H])
    ot = torch.sigmoid(h[:,2*H:3*H])
    gt = torch.tanh(h[:,3*H:4*H])
    
    next_c = torch.multiply(ft,prev_c) + torch.multiply(it,gt)
    next_h = torch.multiply(ot, torch.tanh(next_c))

    return next_h, next_c

def OperatorEmbedding(x,W_embed):
    out = W_embed[x]  
    return out


def affinelayer(x,W_affine,b_affine):
    out = torch.mm(x,W_affine) + b_affine  
    return out


                
def isConstant(symbolic_op,agraph,Node):
    if symbolic_op == 'c':
        agraph.constant_count += 1
        Node.constant_index = agraph.constant_count - 1
        return True
    return False

    
def isLeaf(symbolic_op):
    if symbolic_op == 'c' or symbolic_op == 'x':
        return True, symbolic_op == 'c'
    else:
        return False,False


def samplingLeaf(scores,op_to_idx,deterministic):
    
    leaf_scores = torch.zeros(len(scores))
    leaf_scores[op_to_idx['c']-1] = scores[op_to_idx['c']-1]
    leaf_scores[op_to_idx['x']-1] = scores[op_to_idx['x']-1]
    
    leaf_scores/=torch.sum(leaf_scores)
    if deterministic:
        op = torch.argmax(leaf_scores) + 1
    else:
        scores_np = leaf_scores.detach().cpu().numpy()
        op_array = np.arange(1,len(op_to_idx))
        op = np.random.choice(op_array,p = scores_np)
        op = torch.tensor(op)
    
    return op

def sampling(op_array,op_count,maximum_op,scores,idx_to_op):
    if op_count < maximum_op:
        # Sampling node
        op = np.random.choice(op_array,p = scores)
        op = torch.tensor(op)
    else:
        isleaf,_ =  isLeaf('+')
        while isleaf!=True:
            op = np.random.choice(op_array,p = scores)
            op = torch.tensor(op)
            symbolic_op = idx_to_op[op]
            isleaf,iscon =  isLeaf(symbolic_op)
            
    return op
    

class SymbolicRNN(nn.Module):
    def __init__(self,
                 idx_to_op,op_to_idx,
                 X,y,
                 method,maxiter,threshold,
                 operator_size,embeded_size,hidden_size,
                 device='cpu',dtype=torch.float32):
        
        super().__init__()
        self.maxiter = maxiter
        self.threshold = threshold
        
        self.method = method
        self.device = device
        
        self.idx_to_op = idx_to_op
        self.op_to_idx = op_to_idx
        
        self.X,self.y = X,y
        
        self.D = operator_size
        self.E = embeded_size
        self.H = hidden_size
        
        order =1
        if method == 'LSTM':
            order = 4
        
        self.order = order
        
        
        self.Wx = Parameter(torch.randn(embeded_size,maxiter*hidden_size*order,
                             device=device, dtype=dtype).div(math.sqrt(hidden_size)))
        self.b = Parameter(torch.zeros(maxiter*hidden_size*order,
                             device=device, dtype=dtype))

        self.Wh = Parameter(torch.randn(hidden_size, maxiter*hidden_size*order,
                             device=device, dtype=dtype).div(math.sqrt(hidden_size)))

        self.W_embed = Parameter(torch.randn(operator_size, maxiter*embeded_size,
                             device=device, dtype=dtype).div(math.sqrt(operator_size)))

        self.W_affine = Parameter(torch.randn(hidden_size, maxiter*(operator_size-1),
                             device=device, dtype=dtype).div(math.sqrt(hidden_size)))
        self.b_affine = Parameter(torch.zeros(maxiter*(operator_size-1),
                             device=device, dtype=dtype))
        
        
        
        
        
        
    def forward(self,deterministic=False):
        
        device, maxiter,order =self.device,self.maxiter,self.order
        maxiter_array = [x for x in range(100) if x%2==1]
        maximum_op = maxiter_array.index(maxiter)
        
        # Call empty agraph
        agraph = Agraph()
        
        # Define variable
        N,D,E,H = 2,self.D, self.E, self.H
        op_count = 0
        
        preorder,node_array = [],[]
        
        x0 = torch.tensor([0,0]).to(device)
        h = torch.zeros(N,H).to(device)
        prev_h = h
        if self.method == "LSTM":
            c = torch.zeros(N,H).to(device)
            prev_c = c
            
        p_tau = torch.zeros(maxiter)
        selected_op = torch.zeros(maxiter)
        
        for i in range(maxiter):
            
            # Call weight parameters
            if i==0:
                W_embed = self.W_embed[:,E*i:E*i+E]
                Wx = self.Wx[:,order*H*i:order*(H*i+H)]
                Wh = self.Wh[:,order*H*i:order*(H*i+H)]
                b =self.b[order*H*i:order*(H*i+H)]
                W_affine = self.W_affine[:,(D-1)*i:((D-1)*i)+(D-1)]
                b_affine = self.b_affine[(D-1)*i:((D-1)*i)+(D-1)]
            else:
                
                j = i-1
                W_embed = self.W_embed[:,E*i:E*i+E] + self.W_embed[:,E*j:E*j+E]
                Wx = self.Wx[:,order*H*i:order*(H*i+H)] + self.Wx[:,order*H*j:order*(H*j+H)]
                Wh = self.Wh[:,order*H*i:order*(H*i+H)] + self.Wh[:,order*H*j:order*(H*j+H)] 
                b = self.b[order*H*i:order*(H*i+H)] + self.b[order*H*j:order*(H*j+H)]
                W_affine = self.W_affine[:,(D-1)*i:((D-1)*i)+(D-1)] + self.W_affine[:,(D-1)*j:((D-1)*j)+(D-1)]
                b_affine = self.b_affine[(D-1)*i:((D-1)*i)+(D-1)] + self.b_affine[(D-1)*j:((D-1)*j)+(D-1)]
                
            
            # RNN forward
            x0 = x0.long()
            
            x = OperatorEmbedding(x0,W_embed)
            if self.method == "LSTM":
                
                next_h,next_c = lstm_step_forward(x, prev_h, prev_c, 
                                                  Wx,
                                                  Wh, 
                                                  b)
            else:
                
                next_h = rnn_step_forward(x, prev_h, 
                                          Wx,
                                          b,
                                          Wh)
                
        
            y = affinelayer(next_h,
                            W_affine,
                            b_affine)
            
            y = torch.mean(y,axis=0)
           
            softmax = nn.Softmax(dim=0)
            scores = softmax(y)
            
                
            # convert score to numpy array
            scores_np = scores.detach().cpu().numpy()
            op_array = np.arange(1,len(self.idx_to_op))
            
            # Select operator
            if op_count < maximum_op:
                # Sampling node
                op = np.random.choice(op_array,p = scores_np)
                op = torch.tensor(op)
            else:
                scores_np[self.op_to_idx['+']-1] = 0
                scores_np[self.op_to_idx['-']-1] = 0
                scores_np[self.op_to_idx['*']-1] = 0
                scores_np[self.op_to_idx['/']-1] = 0
                scores_np/=np.sum(scores_np)
                op = sampling(op_array,op_count,maximum_op,scores_np,self.idx_to_op)
            
            
            if i==0:
                scores_np[self.op_to_idx['c']-1] = 0
                scores_np[self.op_to_idx['x']-1] = 0
                scores_np/=np.sum(scores_np)
                op = sampling(op_array,op_count,maximum_op,scores_np,self.idx_to_op)
                
                
            # Build agraph
            prev_node = None
            if len(preorder)!=0:
                prev_node = preorder[-1]
            
            # Define node
            symbolic_op = self.idx_to_op[op]
            
            is_leaf,is_con = isLeaf(symbolic_op)
            
            # if cur_node is constant and its neighborhood is also constant, change cur_node to variable node
            if is_con and prev_node!=None:
                if prev_node.left!=None:
                    if self.idx_to_op[prev_node.left.op]=='c':
                        scores_np[self.op_to_idx['c']-1] = 0
                        scores_np/=np.sum(scores_np)
                        op = sampling(op_array,op_count,maximum_op,scores_np,self.idx_to_op)
                        symbolic_op = self.idx_to_op[op]
                        
                        
            is_leaf,is_con = isLeaf(symbolic_op)
            if is_leaf == False:
                op_count += 1
           
            # Define node
            cur_node = binary_node(op)
            cur_node.idx = i
            
            # if cur_node is constant, constant count += 1
            is_con = isConstant(symbolic_op,agraph,cur_node)
            cur_node.scores = scores
            node_array.append(cur_node)
            if i==0:
                agraph.root = cur_node
                x0 = torch.tensor([op,0]).to(device)
                preorder.append(cur_node)
            else:
                if prev_node.left == None:
                    prev_node.left = cur_node
                else:
                    prev_node.right = cur_node
                    preorder.pop()
                
                if is_leaf:
                    x0 = torch.tensor([prev_node.op,prev_node.left.op]).to(device)
                else:
                    x0 = torch.tensor([op,0]).to(device)
                    preorder.append(cur_node)
            
            # Save selected operater and corresponding score
            p_tau[i] = scores[op-1]
            selected_op[i] = op
            
            if len(preorder)==0:
                break
        
        p_tau = p_tau[:i+1]
        log_p_tau_array = torch.log(p_tau)
        
        selected_op = selected_op[:i+1]
        
        
        agraph.p_tau = p_tau
        agraph.preorder = selected_op
        # find conststants
        constants = []
        if agraph.constant_count>0:
            root = agraph.root
            constants = localOptimizer(agraph,root,self.X,self.y,self.idx_to_op)
            defineAgraph(agraph,constants,self.idx_to_op)
        
        
        reward_array = torch.zeros(len(log_p_tau_array))
        for i,root in enumerate(node_array):
            reward_array[i] = reward(root,self.idx_to_op,self.X,self.y,constants)
        
        
        R_y = reward(agraph.root,self.idx_to_op,self.X,self.y,constants)
        
        loss_array = torch.zeros(len(reward_array))
        cost_array = torch.zeros(len(reward_array))
        for i,r in enumerate(reward_array):
            root = node_array[i]
            log_p_tau = log_p_tau_array[i]
            p_t = p_tau[i]
            if r>=self.threshold:
                loss_array[i] = log_p_tau*r
                cost_array[i] = p_t*r
            else:
                loss_array[i] = log_p_tau*r*0.1
                cost_array[i] = p_t*r
         
        loss = torch.mean(loss_array)
        cost = torch.mean(cost_array)
        
        
        return agraph,loss,R_y,cost
            

        
        
        
        
        
        
        
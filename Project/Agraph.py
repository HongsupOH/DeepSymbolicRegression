
class binary_node():
    def __init__(self,op):
        self.op = op
        self.constant_data = None
        self.constant_index = -1
        self.scores = None
        self.idx = None
        
        self.left = None
        self.right = None

class unary_node():
    def __init__(self,op):
        self.op = op
        
        self.score = None
        self.child = None
        self.score = None

class Agraph():
    def __init__(self):
        self.root = None
        self.preorder = None
        self.p_tau = None
        self.constant_count = 0

class binary_node():
    def __init__(self,op):
        self.op = op
        self.constant = 0
        self.left = None
        self.right = None

class unary_node():
    def __init__(self,op):
        self.op = op
        self.child = None

class Tree():
    def __init__(self):
        self.root = None
def stringAgraph(root,idx_to_op):

    if root is None:
        return ''
    
    if root.left is None and root.right is None:
        if idx_to_op[root.op] =='c':
            if root.constant_data==None:
                return 'c_{}'.format(root.constant_index)
            else:
                return '{}'.format(root.constant_data)
        elif idx_to_op[root.op] =='x':
            return 'x'
        else:
            return idx_to_op[root.op]
    else:    
        left_sum = stringAgraph(root.left,idx_to_op)
        right_sum = stringAgraph(root.right,idx_to_op)

    # check which operation to apply
    if idx_to_op[root.op] == '+':
        return '('+left_sum + '+' +right_sum +')'

    elif idx_to_op[root.op] == '-':
        return '('+left_sum + '-' + right_sum+')'

    elif idx_to_op[root.op] == '*':
        return '('+left_sum + '*' + right_sum+')'

    elif idx_to_op[root.op] == '/':
        return '('+left_sum + '/' + right_sum +')'
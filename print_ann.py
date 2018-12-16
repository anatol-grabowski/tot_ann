import numpy as np
import re

def arr_to_strs(arr, format='% .2f'):
    try:
        return [[format % el for el in row] for row in arr]
    except TypeError:
        return [format % el for el in arr]

def columnize_lists(*lists):
    length = len(lists[0][0]) if type(lists[0][0]) != str else len(lists[0])
    for i in range(length):
        max_width = 0
        for l in lists:
            is_deep = type(l[0]) != str
            if is_deep:
                col_widths = map(lambda row: len(row[i]), l)
                max_width = max(max_width, *col_widths)
            else:
                max_width = max(max_width, len(l[i]))

        for l in lists:
            is_deep = type(l[0]) != str
            if is_deep:
                for row in l: row[i] = row[i].rjust(max_width)
            else:
                l[i] = l[i].rjust(max_width)

def add_braces_to_block(block, num_braces):
    block = block.split('\n')
    for i, line in enumerate(block):
        if num_braces == 1:
            if i == 0: line = '[ ' + line
            else:      line = ' ' + line
            if i == len(block)-1: line = line + ']'
            else:                 line = line + ' '
        if num_braces == 2:
            line = '[ ' + line + ']'
            if i == 0: line = '[' + line
            else:      line = ' ' + line
            if i == len(block)-1: line = line + ']'
            else:                 line = line + ' '
        block[i] = line
    return '\n'.join(block)

def strs_to_block(strs, orientation=''):
    if orientation == 'diagonal':
        widths = [len(s) + 1 for s in strs]
        lines = [' ' * sum(widths[:i]) + s for i, s in enumerate(strs)]
        lines = [line.ljust(len(lines[-1])) for line in lines]
        return '\n'.join(lines)

    is_deep = type(strs[0]) != str
    if is_deep:
        return '\n'.join([' '.join(row) for row in strs])
    else:
        return ' '.join(strs)

def surround_block(block, pref='', suff=''):
    block = block.split('\n')
    for i, line in enumerate(block): block[i] = pref + line + suff
    return '\n'.join(block)

def arr_to_block(arr, *kargs):
    return strs_to_block(arr_to_strs(arr), *kargs)

def make_string_blocks(layer, weights=None):
    ls = arr_to_strs(layer)
    if weights is None:
        columnize_lists([ls])
        l_block = '\n'.join(ls)
        return l_block

    ws = arr_to_strs(weights)
    columnize_lists(ls, ws)
    w_lines = [' '.join(w_row) for w_row in ws]
    w_block = '\n'.join(w_lines)
    widths = [len(s) + 1 for s in ls]
    l_lines = [' ' * sum(widths[:i]) + s for i, s in enumerate(ls)]
    l_lines = [line.ljust(len(l_lines[-1])) for line in l_lines]
    l_block = '\n'.join(l_lines)
    return l_block, w_block

def connect_right(s, block, connection_string='  '):
    s = s.split('\n')
    block = block.split('\n')
    offset_y = len(s) - len(block)
    center_line_num = len(block) // 2
    for i, line in enumerate(block):
        if i == center_line_num:
            s[offset_y + i] += connection_string + line
        else:
            s[offset_y + i] += ' ' * len(connection_string) + line
    return '\n'.join(s)

def connect_bottom(s, block, connection_string=''):
    s = s.split('\n')
    block = block.split('\n')
    offset_x = len(s[-1]) - len(block[0])
    if connection_string != '':
        s.append(' ' * offset_x + connection_string.center(len(block[0])))
    for line in block:
        s.append(' ' * offset_x + line)
    return '\n'.join(s)



def stack(stack_mode, s, arr, connection_string='', block_mode=''):
    block = arr_to_block(arr, block_mode)
    if stack_mode == 'h':
        return connect_right(s, block, connection_string)
    return connect_bottom(s, block, connection_string)

def to_string(layers_and_weights):
    s = make_string_blocks(layers_and_weights[0])
    w, b = None, None
    l_prev = layers_and_weights[0]
    for i in range(1, len(layers_and_weights)):
        if i % 3 == 1: w = layers_and_weights[i]; continue
        if i % 3 == 2: b = layers_and_weights[i]; continue
        l = layers_and_weights[i]
        prod = (l_prev * w.T).T
        l_prev = l
        sum = prod.sum(axis=0) + b
        s = stack('v', s, [1], '')
        s = stack('h', s, np.vstack((w, b)), ' -> ')
        s = stack('v', s, prod, 'П')
        s = stack('v', s, b, '')
        s = stack('v', s, sum, 'Σ')
        s = stack('v', s, l, 'A', 'diagonal')
    return s

def to_string_back(layers_and_weights):
    target = layers_and_weights[0]
    out = layers_and_weights[1]
    dE_dout = out - target
    delta = layers_and_weights[2]
    b = layers_and_weights[3]
    w = layers_and_weights[4]
    inp = layers_and_weights[5]
    prod = (inp * w.T).T
    sum = prod.sum(axis=0) + b
    dout_dsum = delta / dE_dout
    s = ''
    s += 'w\n' + arr_to_block(w) + '\n'
    s += '' + arr_to_block(b) + '\n'
    # s += 'prod\n' + arr_to_block(prod) + '\n'
    s += 'delta\n' + arr_to_block(delta) + '\n'
    s += 'dout/dsum\n' + arr_to_block(dout_dsum) + '\n'
    s += 'sum\n' + arr_to_block(sum) + '\n'
    s += 'dE/dout\n' + arr_to_block(dE_dout) + '\n'
    s += 'out\n' + arr_to_block(out) + '\n'
    s += 'target\n' + arr_to_block(target) + '\n'
    return s

    w, b = None, None
    for i in range(5, len(layers_and_weights)):
        if i % 4 == 1: w = layers_and_weights[i]; continue
        if i % 4 == 2: b = layers_and_weights[i]; continue
        if i % 4 == 2: d = layers_and_weights[i]; continue
        l = layers_and_weights[i]
        dsum_dout =

    s = stack('v', s, t, 'MSE')
    return s

def print_forward(*to_string_args):
    s = to_string(*to_string_args)
    print(s)

def print_backward(*to_string_args):
    s = to_string_back(*to_string_args)
    print(s)

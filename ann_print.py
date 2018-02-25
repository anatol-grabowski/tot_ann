import numpy as np

format = '% .4f'

def vec_to_lines(vec):
    strs = [format % el for el in vec]
    return [' '.join(strs)]

def vec_to_lines_diag(vec):
    strs = [format % el for el in vec]
    widths = [len(s) + 1 for s in strs]
    lines = [' ' * sum(widths[:i]) + s  for i, s in enumerate(strs)]
    lines = [line.ljust(len(lines[-1])) for line in lines]
    return lines

def vec_to_lines_col(vec):
    strs = [format % el for el in vec]
    return strs

def mat_to_lines(mat):
    strs = [[format % el for el in row] for row in mat]
    return [' '.join(row) for row in strs]

def layer_to_lines(layer):
    lines = []
    lines.append('w')
    lines.extend(mat_to_lines(layer['w']))
    lines.append('net_in')
    lines.extend(vec_to_lines(layer['net_in']))
    lines.append('out')
    lines.extend(vec_to_lines_diag(layer['out']))
    width = max([len(line) for line in lines])
    lines = [line.center(width) for line in lines]
    return lines

def layer_back_to_lines(layer):
    dE_dout = np.dot(layer['delta_next'], np.array(layer['w_next'])[:-1].T)
    dE_din = dE_dout * layer['dout_din']
    out_len = len(layer['out'])
    out_prev_len = len(layer['out_prev'])
    din_dw = np.repeat(layer['out_prev'], out_len).reshape(out_prev_len, out_len)
    din_dw = np.concatenate([din_dw, [np.ones(out_len)]])
    dE_dw = dE_din * din_dw
    w_fix = -layer['learning_rate'] * dE_dw
    w_fixed = layer['w'] + w_fix
    lines = []
    lines.append('w:=w+Δw')
    lines.extend(mat_to_lines(w_fixed))
    lines.append('Δw=-η⋅∂E/∂w')
    lines.extend(mat_to_lines(w_fix))
    lines.append('∂E/∂w')
    lines.extend(mat_to_lines(dE_dw))
    lines.append('∂in/∂w')
    lines.extend(mat_to_lines(din_dw))
    lines.append('w')
    lines.extend(mat_to_lines(layer['w']))
    lines.append('∂E/∂in=δ')
    lines.extend(vec_to_lines(dE_din))
    lines.append('∂out/∂in')
    lines.extend(vec_to_lines(layer['dout_din']))
    lines.append('net_in')
    lines.extend(vec_to_lines(layer['net_in']))
    lines.append('out')
    lines.extend(vec_to_lines_diag(layer['out']))
    lines.append('')
    lines.append('∂E/∂out')
    lines.extend(vec_to_lines_diag(dE_dout))
    width = max([len(line) for line in lines])
    lines = [line.center(width) for line in lines]
    return lines

def input_to_lines(input):
    lines = []
    width = len(vec_to_lines_col(input)[0])
    lines.append('out'.center(width))
    lines.extend(vec_to_lines_col(input))
    return lines

def stack_right_top(left, right, right_offset):
    len_left, len_right = len(left), len(right)
    i_min, i_max = min(0, right_offset), max(len_left, len_right + right_offset)
    lines = []
    for i in range(i_min, i_max):
        left_line, right_line = ' ' * len(left[0]), ' ' * len(right[0])
        if 0 <= i < len_left:  left_line  = left[i]
        right_max = right_offset + len_right
        if right_offset <= i < right_max: right_line = right[i - right_offset]
        lines.append(left_line + '   ' + right_line)
    return lines

def ann_to_lines(layers):
    lines = input_to_lines(layers[0]['out'])
    line_offset = 0
    for i, layer in enumerate(layers):
        if i == 0:
            continue
        layer_lines = layer_to_lines(layer)
        lines = stack_right_top(lines, layer_lines, line_offset)
        line_offset = len(lines) - len(layer['out']) - 1
    return lines

w = np.array([[0.15, 0.25], [0.20, 0.30], [0.35, 0.35]])
net_in = np.array([0.3775, 0.3925])
out = np.array([0.5933, 0.5969])
input_lines = input_to_lines(out)
layer_lines = layer_to_lines({'w': w, 'net_in': net_in, 'out': out})
ann_lines = stack_right_top(input_lines, layer_lines, -20)
#print('\n'.join(ann_lines))

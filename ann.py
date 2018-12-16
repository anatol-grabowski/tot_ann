import numpy as np
import matplotlib.pyplot as plt

eta = 0.1

num_i = 2
num_h = 3
num_o = 2

i = np.zeros(num_i)
h = np.zeros(num_h)
o = np.zeros(num_o)
t = np.zeros(num_o)

h_w = np.ones((num_i, num_h))
h_b = np.zeros(num_h)
o_w = np.ones((num_h, num_o))
o_b = np.zeros(num_o)

af = lambda x: 1 / (1 + np.e ** -x)
daf_dx = lambda o: af(o) * (1 - af(o))

def pass_forward():
    global h, o
    h = af(np.dot(i, h_w) + h_b)
    o = af(np.dot(h, o_w) + o_b)


def calculate_errors():
    global E, o_delta, h_delta
    E = ((t - o) ** 2).sum() / 2
    o_delta = daf_dx(o) * (o - t)
    h_delta = daf_dx(h) * np.sum(o_delta * o_w)


def back_propagate():
    global o_w, o_b, h_w, h_b
    o_w -= eta * o_delta * h
    o_b -= eta * o_delta
    h_w -= eta * h_delta * i
    h_b -= eta * h_delta

import print_ann
h_b.fill(0.2)
o_b.fill(0.3)
pass_forward()
print_ann.print_forward([i, h_w, h_b, h, o_w, o_b, o])
calculate_errors()
print(o_delta)
print_ann.print_backward([i, h_w, h_b, h_delta, h, o_w, o_b, o_delta, o, t][::-1])

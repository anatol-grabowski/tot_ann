import numpy as np


i = np.array([0.05, 0.10])
h = np.zeros(2)
o = np.zeros(2)
hw = np.array([[0.15, 0.20],
                [0.25, 0.30]])
ow = np.array([[0.40, 0.45],
                [0.50, 0.55]])
hb = 0.35
ob = 0.60

af = lambda x: 1 / (1 + np.e ** -x)
daf = lambda y: y * (1 - y)
h = af(np.dot(hw, i) + hb)
o = af(np.dot(ow, h) + ob)

t = np.array([0.01, 0.99])
oE = ((t - o) ** 2) / 2
E = oE.sum()

dE_do = -(t - o)
do_doi = o * (1 - o)
o_delta = dE_do * do_doi
dE_dw = o_delta * h

dE_dh = -(t - o)
do_doi = o * (1 - o)
o_delta = dE_do * do_doi
dE_dw = o_delta * h



eta = 0.5
ow -= eta * dE_dw.reshape(-1, 1)
hw -= eta * dE_dh.reshape(-1, 1)

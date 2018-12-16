import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

N, D_in, D_h, D_out = 5, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(D_in, D_h).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(D_h, D_out).type(dtype), requires_grad=True)

logistic = lambda net_in: 1 / (1 + np.exp(-net_in))
dlogistic = lambda out: out * (1 - out)

learning_rate = 1e-6
for i in range(500):
    h = x.mm(w1)
    y_pred = h.mm(w2)

    loss = (y_pred - y).pow(2).sum() / len(y_pred)
    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
    
    w1.grad.zero_()
    w2.grad.zero_()

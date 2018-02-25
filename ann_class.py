import numpy as np


class Layer():
    def __init__(self, num_neurons, act_fn=None, dact_fn=None):
        self.act_fn = act_fn
        self.dact_fn = dact_fn
        self.net_in = np.empty(num_neurons)
        self.out = np.empty(num_neurons)
        self.dout_din = np.empty(num_neurons)

    def activate(self):
        self.out = self.act_fn(self.net_in)

    def derive(self):
        self.dout_din = self.dact_fn(self.out)

    def __repr__(self):
        return 'Layer out=' + repr(self.out)
        

class ANN():
    learning_rate = 0.5

    def inputs(self, num_neurons):
        if hasattr(self, 'layers'):
            raise TypeError('Input layer can be created only once')
        layer = Layer(num_neurons)
        self.layers = [layer]

    def layer(self, num_neurons, act_fn, dact_fn):
        if not hasattr(self, 'layers'):
            raise TypeError('Input layer should be created first')
        layer = Layer(num_neurons, act_fn, dact_fn)
        out_prev_len = len(self.layers[-1].out)
        layer.w = np.random.normal(0, 1, size=(out_prev_len + 1, num_neurons))
        self.layers.append(layer)

    def forward_pass(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue
            out_prev = self.layers[i - 1].out
            layer.net_in = np.dot(out_prev, layer.w[:-1]) + layer.w[-1]
            layer.activate()

    def calc_deltas(self):
        for i, layer in reversed(list(enumerate(self.layers))):
            if i == 0:
                continue
            layer.derive()
            if i == len(self.layers) - 1:
                layer.delta = (layer.out - layer.target) * layer.dout_din
                continue
            delta_next = self.layers[i + 1].delta
            w_next = self.layers[i + 1].w
            layer.delta = np.dot(delta_next, w_next[:-1].T) * layer.dout_din

    def update_weights(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue
            out_prev = self.layers[i - 1].out
            out_prev_len = len(out_prev)
            out_len = len(layer.out)
            layer.din_dw = np.repeat(out_prev, out_len).reshape(out_prev_len, out_len)
            layer.dE_dw = np.concatenate([layer.delta * layer.din_dw, [layer.delta]])
            layer.w -= self.learning_rate * layer.dE_dw

    def predict(self, input):
        self.layers[0].out = input
        self.forward_pass()
        return self.layers[-1].out

    def learn(self, input_or_target, target=None):
        if target:
            self.predict(input_or_target)
            self.layers[-1].target = target
        else:
            self.layers[-1].target = input_or_target
        self.calc_deltas()
        self.update_weights()

    def error(self):
        sqE = (self.layers[-1].out - self.layers[-1].target) ** 2
        return sqE.mean()

    def __repr__(self):
        return 'ANN out=' + repr(self.layers[-1].out)


logistic = lambda net_in: 1 / (1 + np.exp(-net_in))
dlogistic = lambda out: out * (1 - out)

def main():
    ann = ANN()
    ann.learning_rate = 0.5
    ann.inputs(2)
    ann.layer(2, logistic, dlogistic)
    ann.layer(2, logistic, dlogistic)
    ann.layers[1].w = np.array([[0.15, 0.25],
                                [0.20, 0.30],
                                [0.35, 0.35]])
    ann.layers[2].w = np.array([[0.40, 0.50],
                                [0.45, 0.55],
                                [0.60, 0.60]])
    ann.layers[0].out = [0.05, 0.10]
    ann.layers[2].target = [0.01, 0.99]
    ann.forward_pass()
    ann.calc_deltas()

    from time import time
    start = time()
    for i in range(1000):
        #ann.learn([0.05, 0.10], [0.01, 0.99])
        pass
    print(time() - start)
    print(ann.error())

    import ann_print

    def layers_for_print(ann):
        layers = ann.layers
        layer_dicts = [{'out': layers[0].out}]
        for i, layer in enumerate(layers):
            if i == 0:
                continue
            attrs = ['w', 'net_in', 'out', 'dout_din']
            layer_dict = {attr: getattr(layer, attr) for attr in attrs}
            layer_dict['out_prev'] = layers[i - 1].out
            layer_dict['learning_rate'] = ann.learning_rate
            if i < len(layers) - 1:
                layer_dict['delta_next'] = layers[i + 1].delta
                layer_dict['w_next'] = layers[i + 1].w
            layer_dicts.append(layer_dict)
        return layer_dicts

    ann_lines = ann_print.ann_to_lines(layers_for_print(ann))
    print('\n'.join(ann_lines))
    l_lines = ann_print.layer_back_to_lines(layers_for_print(ann)[1])
    print('\n'.join(l_lines))

if __name__ == '__main__':
    main()

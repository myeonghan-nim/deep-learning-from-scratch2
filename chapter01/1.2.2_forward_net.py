import numpy as np


# define sigmoid layer with class
class Sigmoid:
    # save parameters
    def __init__(self):
        self.params = []  # there is no learning params, so restore it

    # sigmoid function
    def forward(self, x):
        return 1 / (1 + np.exp(-x))


# define affine layer with class
class Affine:
    # save parameters: W: weight, b: bias
    def __init__(self, W, b):
        self.params = [W, b]

    # affine function with forward
    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


# define affine - sigmoid - affine layers with class
class TwoLayerNet:
    # reset params and create 3 layers
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # reset weight and bias
        W1 = np.random.randn(I, H)  # for now, these params are random
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # create layers
        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]

        # collect all params
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


x = np.random.randn(10, 2)  # input data
model = TwoLayerNet(2, 4, 3)  # model
s = model.predict(x)  # predict result
print(s)

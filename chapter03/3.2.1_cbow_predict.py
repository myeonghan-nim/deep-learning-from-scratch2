from commons.layers import MatMul
import numpy as np

# sample word data
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# init weight
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# set layers
in_layer0 = MatMul(W_in)  # input
in_layer1 = MatMul(W_in)  # input
out_layer = MatMul(W_out)  # output

# forward
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)  # input average
s = out_layer.forward(h)  # score
print(s)

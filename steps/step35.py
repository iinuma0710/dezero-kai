if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dzrkai import Variable
from dzrkai.utils import plot_dot_graph
from dzrkai.functions import tanh


x = Variable(np.array(1.0))
y = tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 5
for i in range(iters):
    gx = x.grad
    gx.name = 'gx' + str(iters + 1)
    gx.backward(create_graph=True)

gx = x.grad
gx.name = 'gx' + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file='step35_{}.png'.format(str(iters + 1)))
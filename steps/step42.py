if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dzrkai import Variable
import dzrkai.functions as F
import matplotlib.pyplot as plt


def predict(x):
    y = F.matmul(x, W) + b
    return y


def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("step42_1.png")

x, y = Variable(x), Variable(y)
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros((1, )))
lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    print("iter ", i, ": ", W, b, loss)

line_x = np.expand_dims(np.linspace(0, 1, 100), 1)
line_y = predict(line_x)
plt.scatter(x.data, y.data, color="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(line_x, line_y.data, color="red")
plt.savefig("step42_2.png")
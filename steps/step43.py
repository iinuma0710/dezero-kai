if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dzrkai import Variable
import dzrkai.functions as F
import matplotlib.pyplot as plt


# def predict(x):
#     y = F.matmul(x, W) + b
#     return y


# def mean_squared_error(x0, x1):
#     diff = x0 - x1
#     return F.sum(diff ** 2) / len(diff)


# データの生成
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) +np.random.rand(100, 1)

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("step43_1.png")

# Sigmoid 関数の描画
sigmoid_x = np.expand_dims(np.linspace(-5, 5, 100), 1)
sigmoid_y = F.sigmoid(sigmoid_x)
plt.cla()
plt.plot(sigmoid_x, sigmoid_y.data)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("step43_2.png")

I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.rand(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.rand(H, O))
b2 = Variable(np.zeros(O))

def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 1000 == 0:
        print("(iter: ", i, ") loss: ", loss)

line_x = np.expand_dims(np.linspace(0, 1, 100), 1)
line_y = predict(line_x)
plt.cla()
plt.scatter(x, y, color="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(line_x, line_y.data, color="red")
plt.savefig("step43_3.png")
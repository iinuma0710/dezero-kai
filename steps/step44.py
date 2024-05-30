if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dzrkai import Variable
import dzrkai.functions as F
import dzrkai.layers as L
import matplotlib.pyplot as plt


# データの生成
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) +np.random.rand(100, 1)

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("step44_1.png")

# Sigmoid 関数の描画
sigmoid_x = np.expand_dims(np.linspace(-5, 5, 100), 1)
sigmoid_y = F.sigmoid(sigmoid_x)
plt.cla()
plt.plot(sigmoid_x, sigmoid_y.data)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("step44_2.png")

# 出力サイズのみを指定して Linear インスタンスを作成
l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    # 各 Linear インスタンスのパラメータの勾配をまとめてリセット
    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    # 各 Linear インスタンスのパラメータを更新
    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print("(iter: ", i, ") loss: ", loss)

line_x = np.expand_dims(np.linspace(0, 1, 100), 1)
line_y = predict(line_x)
plt.cla()
plt.scatter(x, y, color="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(line_x, line_y.data, color="red")
plt.savefig("step44_3.png")
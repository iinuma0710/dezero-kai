if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dzrkai import Variable, Model
import dzrkai.functions as F
import dzrkai.layers as L
import matplotlib.pyplot as plt


# データの生成
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) +np.random.rand(100, 1)

# ハイパーパラメータの設定
lr = 0.2
iters = 10000
hidden_size = 10

# モデルの定義
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
        
    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y
    
# モデルのインスタンス作成
model = TwoLayerNet(hidden_size, 1)
model.plot(x, to_file="step45_1.png")

# 学習の開始
for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print("(iter: ", i, ") loss: ", loss)

line_x = np.expand_dims(np.linspace(0, 1, 100), 1)
line_y = model(line_x)
plt.cla()
plt.scatter(x, y, color="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(line_x, line_y.data, color="red")
plt.savefig("step45_2.png")

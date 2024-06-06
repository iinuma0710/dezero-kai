if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dzrkai import Variable
from dzrkai import optimizers
from dzrkai.models import MLP
import dzrkai.functions as F
import matplotlib.pyplot as plt


# データの生成
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) +np.random.rand(100, 1)

# ハイパーパラメータの設定
lr = 0.2
iters = 10000
hidden_size = 10
    
# モデルと最適化手法のインスタンス作成
model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

# 学習の開始
for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()
    optimizer.update()

    if i % 1000 == 0:
        print("(iter: ", i, ") loss: ", loss)

line_x = np.expand_dims(np.linspace(0, 1, 100), 1)
line_y = model(line_x)
plt.cla()
plt.scatter(x, y, color="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(line_x, line_y.data, color="red")
plt.savefig("step46.png")

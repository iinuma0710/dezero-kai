if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import dzrkai
from dzrkai import optimizers
import dzrkai.functions as F
from dzrkai.models import MLP
import matplotlib.pyplot as plt


# データの生成
x_train, t_train = dzrkai.datasets.get_spiral(train=True)

# データの可視化
for i in range(len(x_train)):
    if t_train[i] == 0:
        marker, color = "x", "red"
    elif t_train[i] == 1:
        marker, color = "o", "blue"
    else:
        marker, color = "^", "green"
    plt.scatter(x_train[i, 0], x_train[i, 1], marker=marker, c=color)
plt.savefig("step48_1.png")

# ハイパーパラメータの設定
lr = 1.0
max_epoch = 300
batch_size = 30
hidden_size = 10

# モデルとオプティマイザの生成
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

# 学習
data_size = len(x_train)
max_iter = math.ceil(data_size / batch_size)
loss_list = []
for epoch in range(max_epoch):
    # データのインデックスをシャッフル
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # ミニバッチとしてデータを取り出す
        batch_index = index[(i * batch_size):((i + 1) * batch_size)]
        batch_x = x_train[batch_index]
        batch_t = t_train[batch_index]

        # 順伝播と逆伝播の計算とパラメータの更新
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    # 学習経過の表示
    avg_loss = sum_loss / data_size
    print("epoch %d: loss %.2f" % (epoch + 1, avg_loss))
    loss_list.append(avg_loss)

plt.cla()
plt.plot(range(max_epoch), loss_list)
plt.xlabel("epoch")
plt.ylabel("avg_loss")
plt.savefig("step48_2.png")

# 決定境界の描画
plt.cla()
h = 0.001
x_min, x_max = x_train[:, 0].min() - .1, x_train[:, 0].max() + .1
y_min, y_max = x_train[:, 1].min() - .1, x_train[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

with dzrkai.no_grad():
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(x_train)):
    c = t_train[i]
    plt.scatter(x_train[i][0], x_train[i][1], s=40,  marker=markers[c], c=colors[c])
plt.savefig("step48_3.png")

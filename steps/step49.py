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

# ハイパーパラメータの設定
lr = 1.0
max_epoch = 300
batch_size = 30
hidden_size = 10

# モデルとオプティマイザ、データセットの生成
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)
train_set = dzrkai.datasets.Spiral()

# 学習
data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)
for epoch in range(max_epoch):
    # データのインデックスをシャッフル
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # ミニバッチとしてデータを取り出す
        batch_index = index[(i * batch_size):((i + 1) * batch_size)]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

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

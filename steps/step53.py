if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import cupy
import numpy as np
import dzrkai
from dzrkai import optimizers
from dzrkai.dataloaders import DataLoader
import dzrkai.functions as F
from dzrkai.models import MLP
import matplotlib.pyplot as plt

# 前処理を行う関数
def preprocess(x):
    x = x.flatten()
    # x = x.astype(np.float32)
    x = x.astype(cupy.float32)
    x /= 255.0
    return x


# ハイパーパラメータの設定
max_epoch = 5
batch_size = 100
hidden_size = 1000

# データセットの読み込みとデータローダの準備
train_set = dzrkai.datasets.MNIST(train=True, transform=preprocess)
train_loader = DataLoader(train_set, batch_size, shuffle=True)

# モデルと最適化手法の準備
model = MLP((hidden_size, 10))
optimizer = optimizers.SGD().setup(model)

# パラメータの読み込み
if os.path.exists('step53.npz'):
    model.load_weights('step53.npz')

# GPU にモデルを送る
if dzrkai.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

# 学習
for epoch in range(max_epoch):
    start = time.time()

    sum_loss = 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    # 学習経過の表示
    elapsed_time = time.time() - start
    avg_loss = sum_loss / len(train_set)
    print("epoch {}, loss: {:.4f}, time: {:.4f}[sec]".format(epoch + 1, avg_loss,  elapsed_time))

model.save_weights('step53.npz')
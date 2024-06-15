if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


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
    x = x.astype(np.float32)
    x /= 255.0
    return x


# ハイパーパラメータの設定
max_epoch = 5
batch_size = 100
hidden_size = 1000

# データセットの読み込みとデータローダの準備
train_set = dzrkai.datasets.MNIST(train=True, transform=preprocess)
test_set = dzrkai.datasets.MNIST(train=False, transform=preprocess)
train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

x, t = train_set[0]
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.savefig('step51.png')
print(len(train_set), len(test_set))
print(type(x), x.shape, t)
print('label: ', t)

# モデルと最適化手法の準備
# model = MLP((hidden_size, 10))
model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)

# 学習
for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    # 学習経過の表示
    epoch_loss, epoch_acc = sum_loss / len(train_set), sum_acc / len(train_set)
    print("epoch {}".format(epoch + 1))
    print("train loss {:.4f}, accuracy {:.4f}".format(epoch_loss, epoch_acc))

    # test_set で精度の確認
    sum_loss, sum_acc = 0, 0
    with dzrkai.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    
    # 評価の表示
    epoch_loss, epoch_acc = sum_loss / len(test_set), sum_acc / len(test_set)
    print("test loss {:.4f}, accuracy {:.4f}".format(epoch_loss, epoch_acc))


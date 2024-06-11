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

# ハイパーパラメータの設定
lr = 1.0
max_epoch = 300
batch_size = 30
hidden_size = 10

# データの生成
train_set = dzrkai.datasets.Spiral(train=True)
test_set = dzrkai.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# モデルとオプティマイザの生成
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

# 学習
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
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
    train_loss_list.append(epoch_loss)
    train_acc_list.append(epoch_acc)

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
    epoch_loss, epoch_acc = sum_loss / len(train_set), sum_acc / len(train_set)
    print("test loss {:.4f}, accuracy {:.4f}".format(epoch_loss, epoch_acc))
    test_loss_list.append(epoch_loss)
    test_acc_list.append(epoch_acc)

x = [i for i in range(max_epoch)]
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x, train_loss_list, label="train")
ax1.plot(x, test_loss_list, label="test")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(x, train_acc_list, label="train")
ax2.plot(x, test_acc_list, label="test")
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.legend()

plt.savefig("step50.png")



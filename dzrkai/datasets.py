import gzip
import numpy as np
from matplotlib import pyplot as plt
from dzrkai.utils import get_file


def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=int)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t


class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train

        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index])
        
    def __len__(self):
        return len(self.data)
    
    def prepare(self):
        pass


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)


class MNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, data_dir='/dataset/mnist'):
        self.data_dir = data_dir
        super().__init__(train, transform, target_transform)

    def prepare(self):
        # MNIST のダウンロードサイトの URL
        url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
        
        # 学習セットのファイル
        train_files = {
            'target': 'train-images-idx3-ubyte.gz',
            'label': 'train-labels-idx1-ubyte.gz'
        }
        
        # テストセットのファイル
        test_files = {
            'target': 't10k-images-idx3-ubyte.gz',
            'label': 't10k-labels-idx1-ubyte.gz'
        }
        
        # 画像・ラベルのファイルパスを取得
        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'], self.data_dir)
        label_path = get_file(url + files['label'], self.data_dir)

        # ファイルから画像とラベルを取得
        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_data(self, file_path):
        # 最初の16バイトは飛ばして読みだす
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # 画像サイズ (28x28) に合わせて NCHW にテンソルを整形
        data = data.reshape(-1, 1, 28, 28)
        return data

    def _load_label(self, file_path):
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    
    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

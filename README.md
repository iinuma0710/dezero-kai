# DeZero-Kai: DeZero を改良した深層学習ライブラリ
「[ゼロから作る Deep Learning ➌](https://www.oreilly.co.jp/books/9784873119069/)」で実装されている DeZero をベースに、同シリーズの別冊で紹介されている強化学習や生成モデルなども含めて実装していきます。

## 環境構築
実行環境は WSL 上の Ubuntu に Docker で構築します。
WSL のセットアップや Docker で GPU を使えるようにする方法は、[このあたりの Qiita ページ](https://qiita.com/nabion/items/4c4d4d4119c8586cbd9e) を参考にしてください。
NVIDIA の配布している [CUDA や PyTorch、Jupyter Lab の入ったイメージ](https://hub.docker.com/r/nvidia/cuda/)を使い、```docker compose``` でコンテナの起動などの処理を行います。

```bash
# 設定ファイルの準備
$ mv docker-compose.sample.yml docker-compose.yml

# イメージの用意とコンテナの起動
$ docker compose build
$ docker compose up -d
```

コンテナを起動すると同時に Jupyter Lab も立ち上がるように設定しています。
Jupyter Lab を利用する際には、上記のコマンド実行後、[http://localhost:8888](http://localhost:8888) にアクセスしてください。
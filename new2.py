import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

#----------------------------------------------------------
# ハイパーパラメータなどの設定値
num_epochs = 300         # 学習を繰り返す回数
num_batch = 30         # 一度に処理する画像の枚数
learning_rate = 0.001   # 学習率
input_size = 13         # 入力数

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------------------------------------------------------
# 学習用／評価用のデータセットの作成

# 変換方法の指定
transform = transforms.Compose([
    transforms.ToTensor()
    ])

# MNISTデータの取得
# https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
# 学習用

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
dataset_mean = df.mean()

# Centering
df_c = df-df.mean()
# Normalizing
df_norm = (df_c - df_c.mean()) / (df_c.max() - df_c.min())

# Split data
from keras.utils import np_utils
train_X, test_X, train_Y, test_Y = train_test_split(df_norm, wine.target[:], test_size = 0.25)

train_X = torch.FloatTensor(train_X.values)
train_Y = torch.LongTensor(train_Y)
test_X = torch.FloatTensor(test_X.values)
test_Y = torch.LongTensor(test_Y)

train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
# 評価用
test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)

# データローダー
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = num_batch,
    shuffle = True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,     
    batch_size = num_batch,
    shuffle = True)

#----------------------------------------------------------
# ニューラルネットワークモデルの定義
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 各クラスのインスタンス（入出力サイズなどの設定）
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#----------------------------------------------------------
# ニューラルネットワークの生成
model = Net(input_size, 3).to(device)

#----------------------------------------------------------
# 損失関数の設定
criterion = nn.CrossEntropyLoss() 

#----------------------------------------------------------
# 最適化手法の設定
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

#----------------------------------------------------------
# 学習
model.train()  # モデルを訓練モードにする

save_loss = 10000.0
for epoch in range(num_epochs): # 学習を繰り返し行う
    loss_sum = 0

    for inputs, labels in train_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # optimizerを初期化
        optimizer.zero_grad()

        # ニューラルネットワークの処理を行う
        # inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss = criterion(outputs, labels)
        loss_sum += loss

        # 勾配の計算
        loss.backward()

        # 重みの更新
        optimizer.step()

    # 学習状況の表示
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")

    # モデルの重みの保存
    torch.save(model.state_dict(), 'model_weights.pth')

#----------------------------------------------------------
# 評価
model.eval()  # モデルを評価モードにする

loss_sum = 0
correct = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ニューラルネットワークの処理を行う
        # inputs = inputs.view(-1, input_size) # 画像データ部分を一次元へ並び変える
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss_sum += criterion(outputs, labels)

        # 正解の値を取得
        pred = outputs.argmax(1)
        # 正解数をカウント
        correct += pred.eq(labels.view_as(pred)).sum().item()

print(f"Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")
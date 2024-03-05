import torch
from torch.utils.data import Dataset
import os
import random
from PIL import Image
from dataset import MusicTrainDatasets, MusicTestDatasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from torchvision.models import efficientnet_v2_s
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from collections import Counter

# Dir_Path
dataset_path = "/chess/project/project1/music/MER_audio_taffc_dataset_wav/spec/"
os.makedirs('../result', exist_ok=True)
os.makedirs('../model', exist_ok=True)

# ハイパーパラメータ
batch_size = 32
learning_rate = 0.001
num_epochs = 50

# データセットの読み込みと前処理
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_datasets = MusicTrainDatasets(dataset_path, transform=transform)
test_datasets = MusicTestDatasets(dataset_path, transform=transform)

train_loader = DataLoader(dataset = train_datasets, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = test_datasets, batch_size=batch_size, shuffle=False)

# モデルの構築
model = efficientnet_v2_s(weights='IMAGENET1K_V1')  
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)  # 新しいクラス数に変更


# デバイスの指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# モデルをGPUに移動
model.to(device)

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練およびテストの損失と精度を記録するリスト
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

best_test_accuracy = 0.0  # 最高のテスト精度を保存する変数
best_epoch = 0  # 最高のテスト精度を達成したエポックを保存する変数

# 学習のループ
for epoch in range(num_epochs):
    # 訓練モード
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    train_loss /= len(train_loader)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)

    # テストモード
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Testing", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    test_loss /= len(valid_loader)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_accuracy)

    # 結果の表示
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
    

    # 最高のテスト精度を持つモデルを保存
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_epoch = epoch
        torch.save(model.state_dict(), '../model/Best_EfficientnetV2.pth')

# 最終的な結果の表示
print(f"Best Test Accuracy of {best_test_accuracy:.2f}% achieved at Epoch {best_epoch + 1}. Model saved.")

# グラフの表示
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.title('Losses')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.title('Accuracies')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('../result/training_results.png')
plt.show()
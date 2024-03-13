import torch
from torch.utils.data import Dataset
import os
import random
import re
from PIL import Image

class MusicTrainDatasets(Dataset):
    def __init__(self, directory=None, transform=None):
        self.directory = directory
        self.transform = transform
        self.img_path_and_label = self.ImgPathAndLabel()

    def __len__(self):
        return len(self.img_path_and_label)

    def __getitem__(self, index):
        img_path, label = self.img_path_and_label[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        return img, label

    def ImgPathAndLabel(self):
        img_path_and_labels = []

        for class_name in ['Q1', 'Q2', 'Q3', 'Q4']:
            class_num = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}[class_name]
            class_dir_raw = os.path.join(self.directory, class_name, 'raw')
            class_dir_noisy = os.path.join(self.directory, class_name, 'noisy')

            for class_dir in [class_dir_raw, class_dir_noisy]:
                if os.path.exists(class_dir):
                    files = [file for file in os.listdir(class_dir) if file.endswith(".png")]

                    # 前半80%の割合でファイルを取得
                    num_files = int(0.8 * len(files))
                    selected_files = files[:num_files]

                    for file in selected_files:
                        image_path = os.path.join(class_dir, file)
                        image_path_and_label = image_path, class_num
                        img_path_and_labels.append(image_path_and_label)

        return img_path_and_labels

class MusicTestDatasets(Dataset):
    def __init__(self, directory=None, transform=None):
        self.directory = directory
        self.transform = transform
        self.img_path_and_label = self.ImgPathAndLabel()

    def __len__(self):
        return len(self.img_path_and_label)

    def __getitem__(self, index):
        img_path, label = self.img_path_and_label[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        return img, label

    def ImgPathAndLabel(self):
        img_path_and_labels = []

        for class_name in ['Q1', 'Q2', 'Q3', 'Q4']:
            class_num = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}[class_name]
            class_dir_raw = os.path.join(self.directory, class_name, 'raw')
            class_dir_noisy = os.path.join(self.directory, class_name, 'noisy')

            for class_dir in [class_dir_raw, class_dir_noisy]:
                if os.path.exists(class_dir):
                    files = [file for file in os.listdir(class_dir) if file.endswith(".png")]

                    # 前半80%の割合でファイルを取得
                    num_files = int(0.8 * len(files))
                    selected_files = files[num_files:]

                    for file in selected_files:
                        image_path = os.path.join(class_dir, file)
                        image_path_and_label = image_path, class_num
                        img_path_and_labels.append(image_path_and_label)

        return img_path_and_labels

# class MusicUnknownDatasets(Dataset):
#     def __init__(self, directory=None, transform=None):
#         self.directory = directory
#         self.transform = transform
#         self.img_path_and_label = self.ImgPathAndLabel()

#     def __len__(self):
#         return len(self.img_path_and_label)

#     def __getitem__(self, index):
#         img_path, label = self.img_path_and_label[index]
#         img = Image.open(img_path).convert('RGB')

#         if self.transform:
#             img = self.transform(img)

#         return img, label

#     def ImgPathAndLabel(self):
#         img_path_and_labels = []
#         for root, dirs, files in os.walk(self.directory):
#             dirs_list = re.split("[/]", root)  # ["/"] でスライス
#             if dirs_list[-1] == 'unknown':  # unknown フォルダの場合
#                 for file in files:
#                     if file.startswith("Q1.") or file.startswith("Q2.") or file.startswith("Q3.") or file.startswith("Q4."):
#                         if file.endswith(".png"):
#                             class_num = {'Q1.': 0, 'Q2.': 1, 'Q3.': 2, 'Q4.': 3}[file[:3]]
#                             image_path = os.path.join(root, file)
#                             image_path_and_label = image_path, class_num
#                             img_path_and_labels.append(image_path_and_label)

#         return img_path_and_labels
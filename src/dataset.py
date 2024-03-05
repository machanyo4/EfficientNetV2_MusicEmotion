import torch
from torch.utils.data import Dataset
import os
import random
import re
from PIL import Image

class MusicTrainDatasets(Dataset):
    def __init__(self, directory = None, transform = None):
        
        self.directory = directory
        self.transform = transform
        # self.label, self.label_to_index = self.findClasses()
        self.img_path_and_label = self.ImgPathAndLabel()

    def __len__(self):
        return len(self.img_path_and_label)

    def __getitem__(self, index):
        img_path, label = self.img_path_and_label[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        return img, label

    # def findClasses(self):
    #     '''class name and class num'''
    #     classes = [d.name for d in os.scandir(self.directory) if d.is_dir() and "Q" in d.name]
    #     class_to_index = {class_name: i for i, class_name in enumerate(classes)}
    #     return classes, class_to_index

    def ImgPathAndLabel(self):
        '''Image path and class num'''
        img_path_and_labels = []
        for root, dirs, files in os.walk(self.directory):
            dirs_list = re.split("[/]", root) # ["/"] でスライス
            if dirs_list[-1] in ['Q1', 'Q2', 'Q3', 'Q4']:
                class_num = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}[dirs_list[-1]]
                # 前半80%の割合でファイルを取得
                num_files = int(0.8 * len(files))
                selected_files = files[:num_files]
                for file in selected_files:
                    if file.endswith(".png"):
                        image_path = os.path.join(root, file)
                        image_path_and_label = image_path, class_num
                        img_path_and_labels.append(image_path_and_label)

        return img_path_and_labels

class MusicTestDatasets(Dataset):
    def __init__(self, directory = None, transform = None):
        
        self.directory = directory
        self.transform = transform
        # self.label, self.label_to_index = self.findClasses()
        self.img_path_and_label = self.ImgPathAndLabel()

    def __len__(self):
        return len(self.img_path_and_label)

    def __getitem__(self, index):
        img_path, label = self.img_path_and_label[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        return img, label

    # def findClasses(self):
    #     '''class name and class num'''
    #     classes = [d.name for d in os.scandir(self.directory) if d.is_dir() and "Q" in d.name]
    #     class_to_index = {class_name: i for i, class_name in enumerate(classes)}
    #     return classes, class_to_index

    def ImgPathAndLabel(self):
        '''Image path and class num'''
        img_path_and_labels = []
        for root, dirs, files in os.walk(self.directory):
            dirs_list = re.split("[/]", root) # ["/"] でスライス
            if dirs_list[-1] in ['Q1', 'Q2', 'Q3', 'Q4']:
                class_num = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}[dirs_list[-1]]
                # 後半20%の割合でファイルを取得
                num_files = int(0.8 * len(files))
                selected_files = files[num_files:]
                for file in selected_files:
                    if file.endswith(".png"):
                        image_path = os.path.join(root, file)
                        image_path_and_label = image_path, class_num
                        img_path_and_labels.append(image_path_and_label)

        return img_path_and_labels

class MusicUnknownDatasets(Dataset):
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
        for root, dirs, files in os.walk(self.directory):
            dirs_list = re.split("[/]", root)  # ["/"] でスライス
            if dirs_list[-1] == 'unknown':  # unknown フォルダの場合
                for file in files:
                    if file.startswith("Q1.") or file.startswith("Q2.") or file.startswith("Q3.") or file.startswith("Q4."):
                        if file.endswith(".png"):
                            class_num = {'Q1.': 0, 'Q2.': 1, 'Q3.': 2, 'Q4.': 3}[file[:3]]
                            image_path = os.path.join(root, file)
                            image_path_and_label = image_path, class_num
                            img_path_and_labels.append(image_path_and_label)

        return img_path_and_labels
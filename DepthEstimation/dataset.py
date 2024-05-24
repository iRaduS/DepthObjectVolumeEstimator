import os
import pandas as pd
from torch.utils.data import dataset, DataLoader
from torchvision import transforms
from PIL import Image


class DepthEstimationDataset(dataset.Dataset):
    def __init__(self, trainable=True):
        super().__init__()

        self.trainable = trainable
        self.path = './datasets/DepthEstimationDataset/'
        self.dataset_used = os.path.join('./datasets/DepthEstimationDataset/',
                                         'nyu2_train.csv' if self.trainable else 'nyu2_test.csv')
        self.data = pd.read_csv(filepath_or_buffer=self.dataset_used, header=None)
        self.images = self.data.iloc[:, 0].apply(lambda data: f'{self.path}{"/".join(data.split("/")[1:])}')
        self.depth_maps = self.data.iloc[:, 1].apply(lambda data: f'{self.path}{"/".join(data.split("/")[1:])}')

        self.train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        self.test_transformer = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, depth_map = Image.open(self.images[index]), Image.open(self.depth_maps[index])

        if self.trainable is True:
            image, depth_map = self.train_transformer(image), self.train_transformer(depth_map)
        image, depth_map = self.test_transformer(image), self.test_transformer(depth_map)

        return image, depth_map


def return_train_dataloader(batch_size):
    train_dataset = DepthEstimationDataset()
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def return_validation_dataloader(batch_size):
    valid_dataset = DepthEstimationDataset(trainable=False)
    return DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

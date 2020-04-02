import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing

from data_handling import utils
from data_handling.base import BaseDataset


"""
This class contains an example class for the CIFAR10 Dataset.
"""


class ExampleCIFAR10Dataset(BaseDataset):
    def __init__(self, images_path, has_labels, label_path=None):
        self.images_list = np.asarray(
            utils.list_files_in_dir(images_path, '.png'))
        self.images_path = images_path
        self.has_labels = has_labels
        self.classes = ('airplane', 'automobile', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.label_encoder = preprocessing.LabelEncoder().fit(self.classes)
        if self.has_labels:
            self.labels = pd.read_csv(label_path, index_col=0)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, item):
        image_name = self.images_list[item]
        image = np.asarray(
            Image.open(os.path.join(self.images_path, image_name)))
        image = self.transform_input(image)
        id = int(image_name[:-4])
        if self.has_labels:
            label = self.labels.loc[id, 'label']
            label = self.transform_label(label)
            return {'Input': image, 'Label': label, 'Id': id}
        else:
            return {'Input': image, 'Id': id}

    def transform_input(self, image):
        image = image / 255.0
        image = \
            image - np.mean(image, axis=(0, 1)) / np.std(image, axis=(0, 1))
        return np.transpose(image, [2, 1, 0])

    def transform_label(self, label):
        return self.label_encoder.transform([label])[0]

    def inverse_transform_label(self, one_hot_labels):
        return self.label_encoder.inverse_transform(one_hot_labels)

"""Pytorch Dataset object that loads 27x27 patches that contain single cells."""

import os
import random
import scipy.io
import numpy as np
from skimage import io, color
import glob
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms

import dataloaders.utils_augmentation as utils_augmentation


class BreastCancerBagsCross(data_utils.Dataset):
    def __init__(self, path,train_val_idxs=None, test_idxs=None, train=True, shuffle_bag=False,
                 data_augmentation=False, loc_info=False):
        self.path = path
        self.train_val_idxs = train_val_idxs
        self.test_idxs = test_idxs
        self.train = train
        self.shuffle_bag = shuffle_bag
        self.data_augmentation = data_augmentation
        self.location_info = loc_info
        self.stride=16

        self.data_augmentation_img_transform = transforms.Compose([utils_augmentation.RandomHEStain(),
                                                                   utils_augmentation.RandomRotate(),
                                                                   utils_augmentation.RandomVerticalFlip(),
                                                                   transforms.RandomHorizontalFlip(),
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize((0, 0, 0),
                                                                                        (1, 1, 1))
                                                                   ])

        self.normalize_to_tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0),
                                 (1, 1, 1))
        ])

        self.dir_list_train, self.dir_list_test = self.split_dir_list(self.path, self.train_val_idxs, self.test_idxs)
        if self.train:
            self.bag_list_train, self.labels_list_train = self.create_bags(self.dir_list_train)
        else:
            self.bag_list_test, self.labels_list_test = self.create_bags(self.dir_list_test)

    @staticmethod
    def split_dir_list(path, train_val_idxs, test_idxs):

        dirs = glob.glob(path+"/*.tif")


        dirs.pop(0)
        dirs.sort()

        dir_list_train = [dirs[i] for i in train_val_idxs]
        dir_list_test = [dirs[i] for i in test_idxs]

        return dir_list_train, dir_list_test

    def create_bags(self, dir_list):
        bag_list = []
        labels_list = []
        for dir in dir_list:
            # Get image name
            img_name = dir.split('/')[-1]

            # bmp to pillow
            img_dir = self.path + '/' + img_name


            img = io.imread(img_dir)
            if img.shape[2] == 4:
                img = color.rgba2rgb(img)

            if self.location_info:
                xs = np.arange(0, 500)
                xs = np.asarray([xs for i in range(500)])
                ys = xs.transpose()
                img = np.dstack((img, xs, ys))

            cropped_cells = []
            labels = []
            # crop cells
            label = str("0") if "benign" in img_name else str("1")

            temp_name= os.path.splitext(img_name)[0]
            with open(os.path.join(self.path, "img{}.txt".format(temp_name)), "r") as cell_loc:
                lines = cell_loc.readlines()

                for line in lines:

                    x=line.split(",")[0]
                    y= line.split(",")[1]
                    patch = img[int(x) - self.stride:int(x) + self.stride,
                       int(y) - self.stride:int(y) + self.stride]

                    cropped_cells.append(patch)
                    labels.append(int(label))

                # generate bag
            bag = cropped_cells

            # store single cell labels
            labels = np.array(labels)


            if self.shuffle_bag:
                zip_bag_labels = list(zip(bag, labels))
                random.shuffle(zip_bag_labels)
                bag, labels = zip(*zip_bag_labels)


            if self.train:
                bag_list.append(bag)
                labels_list.append(labels)
            else:
                bag_list.append(bag)
                labels_list.append(labels)

        return bag_list, labels_list

    def transform_and_data_augmentation(self, bag):
        if self.data_augmentation:
            img_transform = self.data_augmentation_img_transform

        else:
            img_transform = self.normalize_to_tensor_transform

        bag_tensors = []
        for img in bag:

            if self.location_info:
                bag_tensors.append(torch.cat(
                    (img_transform(img[:, :, :3]),
                     torch.from_numpy(img[:, :, 3:].astype(float).transpose((2, 0, 1))).float())))
            else:
                bag_tensors.append(img_transform(img))

        return torch.stack(bag_tensors)

    def __len__(self):
        if self.train:
            return len(self.labels_list_train)
        else:
            return len(self.labels_list_test)

    def __getitem__(self, index):
        if self.train:
                bag = self.bag_list_train[index]
                label = self.labels_list_train[index]
        else:
                bag = self.bag_list_test[index]
                label = self.labels_list_test[index]
        return self.transform_and_data_augmentation(bag), label

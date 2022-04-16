import os
import random
from typing import final
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import pydicom
import utils

from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader

path = os.path.dirname(os.path.abspath(__file__))

# hyper-parameter
size_width = 256
size_height = 256
do_img_augment = False

# augmentation methods and codes
methods = [None, 'flip'] # TODO: add more methods
codebook = {
    'flip': [-1, 0, 1],
}

class FeatureDataset(Dataset):
    def __init__(self, is_train = True, output_ori_image = False):
        self.is_train = is_train
        self.output_ori_image = output_ori_image
        self.images = []
        self.labels = []
        self.image_size = []

        if self.is_train:
            self.getAllDataPath(f'{path}/CHAOS_AIAdatasets/1_Domain_Gernalization_dataset/Train_Sets')
            self.labels.sort()
        else:
            self.getAllDataPath(f'{path}/CHAOS_AIAdatasets/1_Domain_Gernalization_dataset/Test_Sets')

        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data_path = self.images[idx]
        data = self.getImg(data_path)

        label = data_path # for test
        if self.is_train:
            label_path = self.labels[idx]
            label = self.getImg(label_path)

            # print(data_path, label_path) # for check mapping between data and label

            if do_img_augment:
                # do image augmentation
                method = random.choice(methods)
                if None != method:
                    code = random.choice(codebook[method])
                    data = self.imgAugment(data, method, code)
                    label = self.imgAugment(label, method, code)

        return data, label

    def getAllDataPath(self, dir_path):
        for root, dirs, files in os.walk(os.path.abspath(dir_path)):
            mr_ground_lst = []
            for file in sorted(files):
                if 'T1DUAL' in root:
                    continue

                if '.dcm' in file:
                    self.images.append(os.path.join(root, file))
                elif ('.png' in file) and (self.is_train):
                    # if 'T1DUAL' in root:
                    #     # each ground truth of T1 is related to 2 dcm (InPhase & OutPhase)
                    #     mr_ground_lst.append(os.path.join(root, file))
                    self.labels.append(os.path.join(root, file))

            if 0 != len(mr_ground_lst):
                self.labels += mr_ground_lst

    def getImg(self, img_path):
        image = None

        if '.dcm' in img_path:
            dcm = pydicom.dcmread(img_path)
            pixels = dcm.pixel_array
            # plt.imshow(pixels, cmap = plt.cm.bone)
            # plt.show()
            pixels = pixels.astype(float)
            # pixels = (np.maximum(pixels, 0) / pixels.max()) * 255 # for adjust bright

            # if self.is_train:
            #    pixels = np.pad(pixels, ((size_width - pixels.shape[0]) // 2), self.imgPadding)
            # else:
            self.image_size.append([pixels.shape[0], pixels.shape[1]])
            pixels = resize(pixels, (size_width, size_height), anti_aliasing = True)

            if self.output_ori_image:
                # create folder if not exists
                output_path = img_path
                output_path = output_path.replace('Test_Sets', 'Pred').replace('.dcm', '.png')
                dir = '/'.join((output_path.split('/'))[: -1]) + '/'
                utils.checkFolder(dir)

                # process image
                final_image = np.uint8(pixels)
                final_image = Image.fromarray(final_image)
                final_image.save(output_path)

            # pixels = resize(pixels, (size_width, size_height), anti_aliasing = True)
            image = pixels.reshape(1, pixels.shape[0], pixels.shape[1])
        else:
            color_max = 255

            img = cv2.imread(img_path)
            img = cv2.resize(img, (size_width, size_height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # deal with labels of MR
            if 'MR' in img_path:
                mask = cv2.inRange(img, 55, 70) # liver's pixel range is 55 - 70
                img = cv2.bitwise_and(img, img, mask = mask)
                color_max = 55

            # img = np.pad(img, ((size_width - img.shape[0]) // 2), self.imgPadding)
            image = img.reshape(1, img.shape[0], img.shape[1])

            # deal with pixels of the label image
            if 1 < image.max():
                image = image // color_max

            image = image.astype(float)

        return image

    def imgPadding(self, img, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        img[:pad_width[0]] = pad_value
        img[-pad_width[1]:] = pad_value
        return img

    def imgAugment(self, img, method, code):
        #default
        result = eval(f'cv2.{method}(img, code)')
        return result

    def randomSplit(self, percentage):
        dataset_size = len(self.images)
        indices = list(range(dataset_size))
        split = int(np.floor(percentage * dataset_size))

        # np.random.seed(42)
        np.random.shuffle(indices)

        return [indices[split:], indices[:split]]

    def getOriImgSize(self, idx):
        return self.image_size[idx]

if '__main__' == __name__:
    dataset = FeatureDataset(is_train = True)
    print("Numbers: ", len(dataset))
    train_loader = DataLoader(dataset = dataset, batch_size = 2, shuffle = True)
    for image, label in train_loader:
        print(image.shape)
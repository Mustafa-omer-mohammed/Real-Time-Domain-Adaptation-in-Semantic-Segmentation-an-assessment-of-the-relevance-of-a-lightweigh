import torch
import os
from torchvision import transforms
import glob
from utils import get_Idda_info, get_label_info, RandomCrop, one_hot_it_v11_dice, one_hot_it_v11, augmentation, augmentation_pixel
from PIL import Image
import numpy as np
import random

class IDDA(torch.utils.data.Dataset):
    def __init__(self, images_path, labels_path, info_path, csv_path, scale, loss='dice'):
        """
        Args:
            images_path (string): path to images folder
            labels_path (string): path to labels folder
            info_path (string): path to info json file
            csv_path (string): path to CamVid csv file
            scale (int, int): desired shape of the image
            loss (string): type of loss to use - either 'dice' or 'crossentropy'
        """
        super().__init__()
        self.images = []
        self.labels = []
        self.dataset_info = get_Idda_info(info_path)
        self.shape = scale
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        #loading dictionary for labels translation
        self.toCamVidDict = { 0: [  0, 128, 192], 1:[128,   0,   0], 2:[ 64,   0, 128], 3:[192, 192, 128], 4:[ 64,  64, 128],
        5:[ 64,  64,   0], 6:[128,  64, 128], 7:[  0,   0, 192], 8:[192, 128, 128], 9:[128, 128, 128], 10:[128, 128,   0], 255:[0, 0, 0]}
        self.label_info = get_label_info(csv_path)
        #creating lists of images and labels
        self.images.extend(glob.glob(os.path.join(images_path, '*.jpg')))
        self.images.sort()
        self.labels.extend(glob.glob(os.path.join(labels_path, '*.png')))
        self.labels.sort()
        self.loss = loss

        #transformations pipeline to transform image to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __getitem__(self, index):
        #open image and label
        img = Image.open(self.images[index])
        label = Image.open(self.labels[index]).convert("RGB")

        #resize image and label, then crop them
        scale = random.choice(self.scale)
        scale = (int(self.shape[0] * scale), int(self.shape[1] * scale))

        seed = random.random()
        img = transforms.Resize(scale, Image.BILINEAR)(img)
        img = RandomCrop(self.shape, seed, pad_if_needed=True)(img)
        img = np.array(img)

        label = transforms.Resize(scale, Image.NEAREST)(label)
        label = RandomCrop(self.shape, seed, pad_if_needed=True)(label)
        label = np.array(label)
        
        #translete to CamVid color palette
        label = self.__toCamVid(label)

        #apply augmentation
        # ===================================
        # Image augmentation ## Horizontal Flipping ##
        img, label = augmentation(img, label)
        
        # Pixel augmentation ## Gaussian Blur ##
        if random.randint(0,1) == 1:
              img = augmentation_pixel(img)
        
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        #computing losses
        if self.loss == 'dice':
            # label -> [num_classes, H, W]
            label = one_hot_it_v11_dice(label, self.label_info).astype(np.uint8)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            label = torch.from_numpy(label)

            return img, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
            label = torch.from_numpy(label).long()

            return img, label

    def __len__(self):
        return len(self.images)

    def __toCamVid(self, label_IDDA):
        label_CamVid = np.zeros(label_IDDA.shape, dtype=np.uint8)

        for i in range(len(self.dataset_info['label2camvid'])):
            mask = label_IDDA[:,:,0] == self.dataset_info['label2camvid'][i][0]
            label_CamVid[mask] = self.toCamVidDict[self.dataset_info['label2camvid'][i][1]]
        
        return label_CamVid

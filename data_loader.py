from random import choice
from copy import deepcopy
from PIL import Image

import numpy as np
import torch

from albumentations.core.composition import Compose
from albumentations.augmentations.transforms import RandomCrop, ShiftScaleRotate

class ImagesDS(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, mode='train', channels=[1,2,3,4,5,6]):
        self.records = deepcopy(df).to_records(index=False)
        self.channels = channels
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.transform = Compose([
            ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=180, p=1),
            RandomCrop(height=364, width=364, p=1)
            ])
        
    @staticmethod
    def _load_img(img_path):
        img = np.asarray(Image.open(img_path))
        return img

    @staticmethod
    def _show_imgs(imgs):
        from matplotlib import pyplot as plt
        import rxrx.io as rio
        import cv2
        plt.figure()
        for i, img in enumerate(imgs):
            height, width, _ = img.shape
            img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
            img_rgb = np.array(rio.convert_tensor_to_rgb(img), dtype='uint8')
            img_rgb = cv2.resize(img_rgb, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
            plt.subplot(1, len(imgs), i+1)
            plt.imshow(img_rgb)
        plt.show()

    def _transform(self, img):
        img = self.transform(image=img)['image']    
        return img

    def _get_img_path(self, index, channel, site):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir, self.mode, experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.png'])
        
    def __getitem__(self, index):
        paths_site_1 = [self._get_img_path(index, ch, site=1) for ch in self.channels]
        paths_site_2 = [self._get_img_path(index, ch, site=2) for ch in self.channels]

        img_site_1 = np.stack([self._load_img(img_path) for img_path in paths_site_1], axis=2)
        img_site_1_transformed = self._transform(img_site_1)
        # self._show_imgs([img_site_1, img_site_1_transformed])

        img_site_2 = np.stack([self._load_img(img_path) for img_path in paths_site_2], axis=2)
        img_site_2_transformed = self._transform(img_site_2)
        # self._show_imgs([img_site_2, img_site_2_transformed])

        img_site_1_transformed = np.moveaxis(img_site_1_transformed, 2, 0)
        img_site_2_transformed = np.moveaxis(img_site_2_transformed, 2, 0)
        img = torch.Tensor(np.stack([img_site_1_transformed, img_site_2_transformed]))

        if self.mode == 'train':
            return img, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len
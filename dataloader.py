import random
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
import multiprocessing
import os

import pandas as pd
import numpy as np
import cv2
import torch

from albumentations.core.composition import Compose
from albumentations.augmentations.transforms import RandomCrop, ShiftScaleRotate, CenterCrop

class ImagesDS(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, mode, num_workers, channels=[1,2,3,4,5,6]):
        self.records = deepcopy(df).to_records(index=False)
        self.channels = channels
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.transform_train = Compose([
            ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=180, p=1.0),
            RandomCrop(height=364, width=364, p=1.0)
            ], p=1.0)
        self.transform_val = Compose([
            CenterCrop(height=364, width=364, p=1.0)
            ], p=1.0)

        print('Loading images...')
        imgs = list()
        if num_workers < 1:
            num_workers = 1
        pool = multiprocessing.Pool(num_workers)
        pbar = tqdm(total=len(self.records))
        def update(*a):
            pbar.update()
        for i in range(pbar.total):
            imgs.append(pool.apply_async(self._load_imgs, args=(i,), callback=update))
        pool.close()
        pool.join()
        for i in range(len(imgs)):
            imgs[i] = imgs[i].get()

        self.imgs = dict()
        for index in range(len(self.records)):
            experiment, plate, well = self.records[index].experiment, self.records[index].plate, self.records[index].well
            if not(experiment in self.imgs):
                self.imgs[experiment] = dict()
            if not(plate in self.imgs[experiment]):
                self.imgs[experiment][plate] = dict()
            self.imgs[experiment][plate][well] = imgs[index]

    def _get_img_path(self, index, channel, site):
            experiment, plate, well = self.records[index].experiment, self.records[index].plate, self.records[index].well
            if (self.mode == 'train') or (self.mode == 'val'):
                mode = 'train'
            elif self.mode == 'test':
                mode = 'test'
            return '/'.join([self.img_dir, mode, experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.jpeg'])

    def _load_imgs(self, index):
        paths_site_1 = [self._get_img_path(index, ch, site=1) for ch in self.channels]
        paths_site_2 = [self._get_img_path(index, ch, site=2) for ch in self.channels]
        
        img_site_1, img_site_2 = list(), list()
        for img_path in paths_site_1:
            with open(img_path,'rb') as f: 
                img_site_1.append(f.read())
 
        for img_path in paths_site_2:
            with open(img_path,'rb') as f: 
                img_site_2.append(f.read())
 
        return [img_site_1, img_site_2]

    def _show_imgs(self, imgs):
        from matplotlib import pyplot as plt
        import rxrx.io as rio
        import cv2
        fig = plt.figure()
        for i, img in enumerate(imgs):
            height, width, _ = img.shape
            img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
            img_rgb = np.array(rio.convert_tensor_to_rgb(img), dtype='uint8')
            img_rgb = cv2.resize(img_rgb, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
            ax = plt.subplot(1, len(imgs), i+1)
            ax.title.set_text(self.mode)
            plt.imshow(img_rgb)
        plt.show()

    def _transform(self, img):
        if self.mode == 'train':
            img = self.transform_train(image=img)['image']    
        elif (self.mode == 'val'):
            img = self.transform_val(image=img)['image']    
        return img

    def __getitem__(self, index):
        experiment, plate, well = self.records[index].experiment, self.records[index].plate, self.records[index].well
        img_site_1, img_site_2 = self.imgs[experiment][plate][well]

        temp = list()
        for img in img_site_1:
            temp.append(cv2.imdecode(np.frombuffer(img, dtype=np.uint8), -1))
        img_site_1 = np.moveaxis(np.stack(temp), 0, 2)/255

        temp = list()
        for img in img_site_2:
            temp.append(cv2.imdecode(np.frombuffer(img, dtype=np.uint8), -1))
        img_site_2 = np.moveaxis(np.stack(temp), 0, 2)/255

        img_site_1 = self._transform(img_site_1)
        img_site_2 = self._transform(img_site_2)

        # self._show_imgs([img_site_1])
        # self._show_imgs([img_site_2])
        
        img_site_1 = np.moveaxis(img_site_1, 2, 0)
        img_site_2 = np.moveaxis(img_site_2, 2, 0)
        img = torch.Tensor(np.stack([img_site_1, img_site_2]))

        if (self.mode == 'train') or (self.mode == 'val'):
            return img, int(self.records[index].sirna)
        elif (self.mode == 'test'):
            return img, self.records[index].id_code

    def __len__(self):
        return self.len

def train_test_split(df, random_state):
    random.seed(random_state)
    df_train, df_val = list(), list()
    for celltype in df['celltype'].unique():
        df_celltype = df[df['celltype']==celltype]
        experiments = df_celltype['experiment'].unique()
        nb_experiments_val = len(experiments)//3
        random.shuffle(experiments)
        experiments_val = experiments[:nb_experiments_val]
        mask_val = np.zeros(len(df_celltype), dtype=np.uint8)
        for experiment_val in experiments_val:
            mask_val = mask_val + (df_celltype['experiment']==experiment_val)
        mask_val = (mask_val==1)
        mask_train = ~mask_val
        df_celltype_train = df_celltype[mask_train]
        df_celltype_val = df_celltype[mask_val]
        df_train.append(df_celltype_train)
        df_val.append(df_celltype_val)
    df_train = pd.concat(df_train)
    df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_val = pd.concat(df_val)
    df_val = df_val.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df_train, df_val
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
from albumentations.augmentations.transforms import Normalize, RandomCrop, ShiftScaleRotate, CenterCrop

class ImagesDS(torch.utils.data.Dataset):
    def __init__(self, df, df_controls, img_dir, mode, num_workers, channels=[1,2,3,4,5,6]):
        self.records = deepcopy(df).to_records(index=False)
        df_controls = deepcopy(df_controls)
        mask = (df_controls['well_type']=='negative_control') & (df_controls['well']=='B02')
        df_controls = df_controls[mask]
        self.records_controls = df_controls.to_records(index=False)
        self.mode = mode
        self.num_workers = num_workers
        self.channels = channels
        self.img_dir = img_dir
        self.len = df.shape[0]
        mean = (0.02290913, 0.06102184, 0.03960226, 0.03904865, 0.02184808, 0.03553102)
        std = (0.04808127, 0.06136712, 0.0375606, 0.04815974, 0.0472975, 0.03571597)
        self.transform_train = Compose([
            Normalize(mean=mean, std=std, \
                max_pixel_value=255.0, p=1.0),
            ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=180, p=1.0),
            RandomCrop(height=364, width=364, p=1.0)
            ], p=1.0)
        self.transform_val = Compose([
            Normalize(mean=mean, std=std, \
                max_pixel_value=255.0, p=1.0),
            CenterCrop(height=364, width=364, p=1.0)
            ], p=1.0)
        self.transform_test = Compose([
            Normalize(mean=mean, std=std, \
                max_pixel_value=255.0, p=1.0),
            ], p=1.0)

        print('Loading images...')
        self.imgs = self._load_imgs_parallel(self.records)
        print('Loading controls...')
        self.imgs_controls = self._load_imgs_parallel(self.records_controls)

    def _get_img_path(self, records, index, channel, site):
            experiment, plate, well = records[index].experiment, records[index].plate, records[index].well
            if (self.mode == 'train') or (self.mode == 'val'):
                mode = 'train'
            elif self.mode == 'test':
                mode = 'test'
            return '/'.join([self.img_dir, mode, experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.jpeg'])

    def _load_imgs(self, records, index):
        paths_site_1 = [self._get_img_path(records, index, ch, site=1) for ch in self.channels]
        paths_site_2 = [self._get_img_path(records, index, ch, site=2) for ch in self.channels]
        
        img_site_1, img_site_2 = list(), list()
        for img_path in paths_site_1:
            with open(img_path,'rb') as f: 
                img_site_1.append(f.read())
 
        for img_path in paths_site_2:
            with open(img_path,'rb') as f: 
                img_site_2.append(f.read())
 
        return [img_site_1, img_site_2]

    def _load_imgs_parallel(self, records):
        if self.num_workers < 1:
            self.num_workers = 1
        pool = multiprocessing.Pool(self.num_workers)
        pbar = tqdm(total=len(records))
        def update(*a):
            pbar.update()
        imgs = [pool.apply_async(self._load_imgs, args=(records,i,), callback=update) for i in range(pbar.total)]
        pool.close()
        pool.join()
        temp = [imgs[i].get() for i in range(len(imgs))]

        imgs_dict = dict()
        for index in range(len(records)):
            experiment, plate, well = records[index].experiment, records[index].plate, records[index].well
            if not(experiment in imgs_dict):
                imgs_dict[experiment] = dict()
            if not(plate in imgs_dict[experiment]):
                imgs_dict[experiment][plate] = dict()
            imgs_dict[experiment][plate][well] = temp[index]

        return imgs_dict

    def _show_imgs(self, imgs):
        from matplotlib import pyplot as plt
        import rxrx.io as rio
        import cv2
        fig = plt.figure()
        for i, img in enumerate(imgs):
            img = np.moveaxis(img, 0, 2)
            height, width, _ = img.shape
            img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
            img_rgb = np.array(rio.convert_tensor_to_rgb(img), dtype='uint8')
            img_rgb = cv2.resize(img_rgb, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
            ax = plt.subplot(1, len(imgs), i+1)
            ax.title.set_text(self.mode)
            plt.imshow(img_rgb)
        plt.show()

    def _transform(self, img):
        img = np.moveaxis(img, 0, 2)
        if self.mode == 'train':
            img = self.transform_train(image=img)['image']
        elif self.mode == 'val':
            img = self.transform_val(image=img)['image']
        elif self.mode == 'test':
            img = self.transform_test(image=img)['image']
        img = np.moveaxis(img, 2, 0)
        return img

    def _load_from_buffer(self, img_buffer):
        img = list()
        for buffer in img_buffer:
            img.append(cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), -1))
        img = np.stack(img)
        img_transformed = self._transform(img)

        # self._show_imgs([img, img_transformed])

        return img_transformed

    def __getitem__(self, index):
        experiment, plate, well = self.records[index].experiment, self.records[index].plate, self.records[index].well
        img_site_1, img_site_2 = self.imgs[experiment][plate][well]
        img_control_site_1, img_control_site_2 = self.imgs_controls[experiment][plate]['B02']

        img_site_1 = self._load_from_buffer(img_site_1)
        img_site_2 = self._load_from_buffer(img_site_2)
        img_control_site_1 = self._load_from_buffer(img_control_site_1)
        img_control_site_2 = self._load_from_buffer(img_control_site_2)

        img = torch.Tensor(np.stack([img_site_1, img_site_2, img_control_site_1, img_control_site_2]))

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
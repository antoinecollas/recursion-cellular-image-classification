import random
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
import multiprocessing
import os
import random

import pandas as pd
import numpy as np
import cv2
import torch

from albumentations.core.composition import Compose
from albumentations.augmentations.transforms import RandomCrop, ShiftScaleRotate, CenterCrop

class ImagesDS(torch.utils.data.Dataset):
    def __init__(self, df, df_controls, stats_experiments, img_dir, mode, channels=[1,2,3,4,5,6]):
        self.records = deepcopy(df).to_records(index=False)

        df_controls = deepcopy(df_controls)
        mask = (df_controls['well_type']=='negative_control') & (df_controls['well']=='B02')
        df_negative_controls = df_controls[mask]
        self.records_negative_controls = df_negative_controls.to_records(index=False)
        mask = (df_controls['well_type']=='positive_control')
        df_positive_controls = df_controls[mask]
        self.records_positive_controls = df_positive_controls.to_records(index=False)

        self.stats_experiments = stats_experiments
        self.mode = mode
        self.channels = channels
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.transform_train = Compose([
            ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=180, p=1.0),
            RandomCrop(height=364, width=364, p=1.0)
            ], p=1.0)
        self.transform_val = Compose([
            CenterCrop(height=364, width=364, p=1.0)
            ], p=1.0)

        print()
        self.imgs = self._load_imgs(self.records, desc='Images')
        self.imgs_negative_controls = self._load_imgs(self.records_negative_controls, desc='Negative controls')
        self.imgs_positive_controls = self._load_imgs(self.records_positive_controls, desc='Positive controls')

    def _get_img_path(self, records, index, channel, site):
            experiment, plate, well = records[index].experiment, records[index].plate, records[index].well
            if (self.mode == 'train') or (self.mode == 'val'):
                mode = 'train'
            elif self.mode == 'test':
                mode = 'test'
            return '/'.join([self.img_dir, mode, experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.jpeg'])

    def _load_imgs(self, records, desc):
        imgs = list()
        for index in tqdm(range(len(records)), desc=desc):
            paths_site_1 = [self._get_img_path(records, index, ch, site=1) for ch in self.channels]
            paths_site_2 = [self._get_img_path(records, index, ch, site=2) for ch in self.channels]
            
            img_site_1, img_site_2 = list(), list()
            for img_path in paths_site_1:
                with open(img_path,'rb') as f: 
                    img_site_1.append(f.read())
    
            for img_path in paths_site_2:
                with open(img_path,'rb') as f: 
                    img_site_2.append(f.read())
    
            imgs.append([img_site_1, img_site_2])

        imgs_dict = dict()
        for index in range(len(records)):
            experiment, plate, well = records[index].experiment, records[index].plate, records[index].well
            if not(experiment in imgs_dict):
                imgs_dict[experiment] = dict()
            if not(plate in imgs_dict[experiment]):
                imgs_dict[experiment][plate] = dict()
            imgs_dict[experiment][plate][well] = imgs[index]

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

    def _transform(self, img, mean, std):
        img = img / 255.0
        for i in range(img.shape[0]):
            img[i] = (img[i] - mean[i]) / std[i]
        img = np.moveaxis(img, 0, 2)
        if self.mode == 'train':
            img = self.transform_train(image=img)['image']
        elif self.mode == 'val':
            img = self.transform_val(image=img)['image']
        img = np.moveaxis(img, 2, 0)
        return img

    def _load_from_buffer(self, img_buffer):
        img = list()
        for buffer in img_buffer:
            img.append(cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), -1))
        img = np.stack(img)
        return img

    def __getitem__(self, index):
        experiment, plate, well = self.records[index].experiment, self.records[index].plate, self.records[index].well
        mean = self.stats_experiments[experiment]['mean']
        std = self.stats_experiments[experiment]['std']

        if (self.mode == 'train') or (self.mode == 'val'):
            site = random.randint(0, 1)
            temp = self.imgs[experiment][plate][well]
            img = temp[site]
            img = self._load_from_buffer(img)
            img = self._transform(img, mean=mean, std=std)

            site = random.randint(0, 1)
            temp = self.imgs_negative_controls[experiment][plate]['B02']
            img_negative_control = temp[site]
            img_negative_control = self._load_from_buffer(img_negative_control)
            img_negative_control = self._transform(img_negative_control, mean=mean, std=std)

            wells_positive_control = list(self.imgs_positive_controls[experiment][plate].keys())
            well_positive_control = random.sample(wells_positive_control, 1)[0]
            site = random.randint(0, 1)
            temp = self.imgs_positive_controls[experiment][plate][well_positive_control]
            img_positive_control = temp[site]
            img_positive_control = self._load_from_buffer(img_positive_control)
            img_positive_control = self._transform(img_positive_control, mean=mean, std=std)

            # self._show_imgs([img, img_transformed])

            img = torch.Tensor(np.stack([img, img_negative_control, img_positive_control]))

            return img, int(self.records[index].sirna)

        elif (self.mode == 'test'):
            imgs = self.imgs[experiment][plate][well]
            for i in range(len(imgs)):
                imgs[i] = self._load_from_buffer(imgs[i])
                imgs[i] = self._transform(imgs[i], mean=mean, std=std)

            imgs_negative_control = deepcopy(self.imgs_negative_controls[experiment][plate]['B02'])
            for i in range(len(imgs_negative_control)):
                imgs_negative_control[i] = self._load_from_buffer(imgs_negative_control[i])
                imgs_negative_control[i] = self._transform(imgs_negative_control[i], mean=mean, std=std)

            imgs_positive_control = list()
            wells_positive_control = list(self.imgs_positive_controls[experiment][plate].keys())
            for well_positive_control in wells_positive_control:
                imgs_positive_control.append(\
                    self.imgs_positive_controls[experiment][plate][well_positive_control][0])
                imgs_positive_control.append(\
                    self.imgs_positive_controls[experiment][plate][well_positive_control][1])
            for i in range(len(imgs_positive_control)):
                imgs_positive_control[i] = self._load_from_buffer(imgs_positive_control[i])
                imgs_positive_control[i] = self._transform(imgs_positive_control[i], mean=mean, std=std)

            imgs = np.array(imgs)
            imgs_negative_control = np.array(imgs_negative_control)
            imgs_positive_control = np.array(imgs_positive_control)
            imgs = torch.Tensor(np.concatenate([imgs, imgs_negative_control, imgs_positive_control]))

            return imgs, self.records[index].id_code

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
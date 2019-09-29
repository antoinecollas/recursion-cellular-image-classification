import random
from copy import deepcopy
from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2
import torch

from albumentations.core.composition import Compose
from albumentations.augmentations.transforms import \
    RandomCrop, ShiftScaleRotate, CenterCrop, VerticalFlip, HorizontalFlip, \
    Normalize


class ImagesDS(torch.utils.data.Dataset):
    def __init__(self,
                 df,
                 df_controls,
                 stats_experiments,
                 img_dir,
                 mode,
                 verbose=True,
                 channels=[1, 2, 3, 4, 5, 6]):

        self.records = deepcopy(df).to_records(index=False)

        df_conts = deepcopy(df_controls)
        mask = (df_conts['well_type'] == 'negative_cont') & \
               (df_conts['well'] == 'B02')
        df_neg_conts = df_conts[mask]
        self.records_neg_conts = df_neg_conts.to_records(index=False)
        mask = (df_conts['well_type'] == 'positive_cont')
        df_pos_conts = df_conts[mask]
        self.records_pos_conts = df_pos_conts.to_records(index=False)

        self.stats_exps = stats_experiments
        self.mode = mode
        self.channels = channels
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.transform_train = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=180,
                             p=1.0),
            RandomCrop(height=364, width=364, p=1.0)
            ], p=1.0)
        self.transform_val = Compose([
            CenterCrop(height=364, width=364, p=1.0)
            ], p=1.0)

        if verbose:
            print()
        self.imgs = self._load_imgs(self.records, desc='Images',
                                    verbose=verbose)
        self.imgs_neg_conts = self._load_imgs(self.records_neg_conts,
                                              desc='Negative conts',
                                              verbose=verbose)
        self.imgs_pos_conts = self._load_imgs(self.records_pos_conts,
                                              desc='Positive conts',
                                              verbose=verbose)

    def _get_img_path(self, records, index, channel, site):
        exp = records[index].exp
        plate = records[index].plate
        well = records[index].well
        if (self.mode == 'train') or (self.mode == 'val'):
            mode = 'train'
        elif self.mode == 'test':
            mode = 'test'
        return '/'.join([self.img_dir, mode, exp,
                         f'Plate{plate}', f'{well}_s{site}_w{channel}.jpeg'])

    def _load_imgs(self, records, desc, verbose):
        imgs = list()
        if verbose:
            iterator = tqdm(range(len(records)), desc=desc)
        else:
            iterator = range(len(records))
        for index in iterator:
            paths_site_1 = [self._get_img_path(records, index, ch, site=1)
                            for ch in self.channels]
            paths_site_2 = [self._get_img_path(records, index, ch, site=2)
                            for ch in self.channels]

            img_site_1, img_site_2 = list(), list()
            for img_path in paths_site_1:
                with open(img_path, 'rb') as f:
                    img_site_1.append(f.read())

            for img_path in paths_site_2:
                with open(img_path, 'rb') as f:
                    img_site_2.append(f.read())

            imgs.append([img_site_1, img_site_2])

        imgs_dict = dict()
        for index in range(len(records)):
            exp = records[index].exp
            plate = records[index].plate
            well = records[index].well
            if not(exp in imgs_dict):
                imgs_dict[exp] = dict()
            if not(plate in imgs_dict[exp]):
                imgs_dict[exp][plate] = dict()
            imgs_dict[exp][plate][well] = imgs[index]

        return imgs_dict

    def _show_imgs(self, imgs):
        from matplotlib import pyplot as plt
        import rxrx.io as rio
        import cv2
        for i, img in enumerate(imgs):
            img = np.moveaxis(img, 0, 2)
            height, width, _ = img.shape
            img = cv2.resize(img, dsize=(512, 512),
                             interpolation=cv2.INTER_CUBIC)
            img_rgb = np.array(rio.convert_tensor_to_rgb(img), dtype='uint8')
            img_rgb = cv2.resize(img_rgb, dsize=(height, width),
                                 interpolation=cv2.INTER_CUBIC)
            ax = plt.subplot(1, len(imgs), i+1)
            ax.title.set_text(self.mode)
            plt.imshow(img_rgb)
        plt.show()

    def _transform(self, img, mean, std):
        img = np.moveaxis(img, 0, 2)
        if self.mode == 'train':
            img = self.transform_train(image=img)['image']
        elif self.mode == 'val':
            img = self.transform_val(image=img)['image']
        normalize = Compose([
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0)
            ], p=1.0)
        img = normalize(image=img)['image']
        img = np.moveaxis(img, 2, 0)
        return img

    def _load_from_buffer(self, img_buffer):
        img = list()
        for buffer in img_buffer:
            img.append(cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), -1))
        img = np.stack(img)
        return img

    def __getitem__(self, index):
        exp = self.records[index].exp
        plate = self.records[index].plate
        well = self.records[index].well
        mean = self.stats_exps[exp]['mean']
        std = self.stats_exps[exp]['std']

        if (self.mode == 'train') or (self.mode == 'val'):
            site = random.randint(0, 1)
            img = self.imgs[exp][plate][well][site]
            img = self._load_from_buffer(img)
            img = self._transform(img, mean=mean, std=std)

            site = random.randint(0, 1)
            img_neg_cont = self.imgs_neg_conts
            img_neg_cont = img_neg_cont[exp][plate]['B02'][site]
            img_neg_cont = self._load_from_buffer(img_neg_cont)
            img_neg_cont = self._transform(img_neg_cont, mean=mean, std=std)

            wells_pos_cont = list(self.imgs_pos_conts[exp][plate].keys())
            well_pos_cont = random.sample(wells_pos_cont, 1)[0]
            site = random.randint(0, 1)
            img_pos_cont = self.imgs_pos_conts
            img_pos_cont = img_pos_cont[exp][plate][well_pos_cont][site]
            img_pos_cont = self._load_from_buffer(img_pos_cont)
            img_pos_cont = self._transform(img_pos_cont, mean=mean, std=std)

            # self._show_imgs([img, img_transformed])

            img = torch.Tensor(np.stack([img, img_neg_cont,
                                         img_pos_cont]))

            return img, int(self.records[index].sirna)

        elif (self.mode == 'test'):
            imgs = self.imgs[exp][plate][well]
            for i in range(len(imgs)):
                imgs[i] = self._load_from_buffer(imgs[i])
                imgs[i] = self._transform(imgs[i], mean=mean, std=std)

            imgs_neg_cont = deepcopy(self.imgs_neg_conts[exp][plate]['B02'])
            for i in range(len(imgs_neg_cont)):
                imgs_neg_cont[i] = self._load_from_buffer(imgs_neg_cont[i])
                imgs_neg_cont[i] = self._transform(imgs_neg_cont[i],
                                                   mean=mean, std=std)

            wells_pos_cont = list(self.imgs_pos_conts[exp][plate].keys())
            well_pos_cont = random.sample(wells_pos_cont, 1)[0]
            temp = self.imgs_pos_conts[exp][plate][well_pos_cont]
            imgs_pos_cont = deepcopy(temp)
            for i in range(len(imgs_pos_cont)):
                imgs_pos_cont[i] = self._load_from_buffer(imgs_pos_cont[i])
                imgs_pos_cont[i] = self._transform(imgs_pos_cont[i],
                                                   mean=mean, std=std)

            imgs = np.array(imgs)
            imgs_neg_cont = np.array(imgs_neg_cont)
            imgs_pos_cont = np.array(imgs_pos_cont)
            temp = np.concatenate([imgs, imgs_neg_cont, imgs_pos_cont])
            imgs = torch.Tensor(temp)

            return imgs, self.records[index].id_code

    def __len__(self):
        return self.len


def train_test_split(df, random_state):
    random.seed(random_state)
    df_train, df_val = list(), list()
    for celltype in df['celltype'].unique():
        df_celltype = df[df['celltype'] == celltype]
        exps = df_celltype['exp'].unique()
        nb_exps_val = len(exps)//3
        random.shuffle(exps)
        exps_val = exps[:nb_exps_val]
        mask_val = np.zeros(len(df_celltype), dtype=np.uint8)
        for exp_val in exps_val:
            mask_val = mask_val + (df_celltype['exp'] == exp_val)
        mask_val = (mask_val == 1)
        mask_train = ~mask_val
        df_celltype_train = df_celltype[mask_train]
        df_celltype_val = df_celltype[mask_val]
        df_train.append(df_celltype_train)
        df_val.append(df_celltype_val)
    df_train = pd.concat(df_train)
    df_train = df_train.sample(frac=1, random_state=random_state) \
                       .reset_index(drop=True)
    df_val = pd.concat(df_val)
    df_val = df_val.sample(frac=1, random_state=random_state) \
                   .reset_index(drop=True)
    return df_train, df_val

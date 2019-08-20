from random import choice
from copy import deepcopy
from PIL import Image

import torch
import torch.utils.data as D
from torchvision import transforms as T

class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir, mode='train', channels=[1,2,3,4,5,6]):
        self.records = deepcopy(df).to_records(index=False)
        self.channels = channels
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        
    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return T.ToTensor()(img)

    def _get_img_path(self, index, channel, site):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir, self.mode, experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.png'])
        
    def __getitem__(self, index):
        paths_site_1 = [self._get_img_path(index, ch, site=1) for ch in self.channels]
        paths_site_2 = [self._get_img_path(index, ch, site=2) for ch in self.channels]
        img_site_1 = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths_site_1])
        img_site_2 = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths_site_2])
        img = torch.stack([img_site_1, img_site_2])

        if self.mode == 'train':
            return img, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len
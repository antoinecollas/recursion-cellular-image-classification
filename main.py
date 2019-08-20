import os
import datetime
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from data_loader import ImagesDS
from two_sites_nn import TwoSitesNN

from train import train
from test import test

torch.manual_seed(0)

experiment_id = str(datetime.datetime.now().time()).replace(':', '-').split('.')[0]

parser = argparse.ArgumentParser(description='My parser')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--pretrain', default=False, action='store_true')
parser.add_argument('--scheduler', default=False, action='store_true')

args = parser.parse_args()

debug = args.debug
lr = args.lr
pretrain = args.pretrain
scheduler = args.scheduler

if debug:
    PATH_DATA = 'data/samples'
    BATCH_SIZE = 3
else:
    PATH_DATA = 'data'
    BATCH_SIZE = 40
    
PATH_METADATA = os.path.join(PATH_DATA, 'metadata')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    num_workers = os.cpu_count()
else:
    num_workers = 4*torch.cuda.device_count()

if torch.cuda.is_available():
    BATCH_SIZE = BATCH_SIZE*torch.cuda.device_count()

print('Number of workers used:', num_workers, '/', os.cpu_count())
print('Number of GPUs used:', torch.cuda.device_count())

df = pd.read_csv(PATH_METADATA+'/train.csv')
df_train, df_val = train_test_split(df, test_size = 0.1, random_state=42)
df_test = pd.read_csv(PATH_METADATA+'/test.csv')
print('Size training dataset: {}'.format(len(df_train)))
print('Size validation dataset: {}'.format(len(df_val)))
print('Size test dataset: {}\n'.format(len(df_test)))

ds_train = ImagesDS(df=df_train, img_dir=PATH_DATA, mode='train')
ds_val = ImagesDS(df=df_val, img_dir=PATH_DATA, mode='train')
ds_test = ImagesDS(df=df_test, img_dir=PATH_DATA, mode='test')

nb_classes = 1108
model = TwoSitesNN(pretrained=pretrain, nb_classes=nb_classes)
model = torch.nn.DataParallel(model)

train(experiment_id, ds_train, ds_val, model, BATCH_SIZE, lr, scheduler, num_workers, device, debug)
test(experiment_id, df_test, ds_test, model, BATCH_SIZE, lr, scheduler, num_workers, device, debug)
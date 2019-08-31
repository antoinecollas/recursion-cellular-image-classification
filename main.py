import os
import datetime
import argparse
from copy import deepcopy
import sys

import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from dataloader import train_test_split, ImagesDS
from models import TwoSitesNN, DummyClassifier

from train import train
from test import test

import warnings
warnings.filterwarnings('ignore')

# torch.manual_seed(0)

parser = argparse.ArgumentParser(description='My parser')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--experiment_id')
parser.add_argument('--loss', choices=['softmax', 'arcface'], default='softmax')
parser.add_argument('--lr', type=float)
parser.add_argument('--train', default=False, action='store_true')

args = parser.parse_args()

debug = args.debug
experiment_id = args.experiment_id
loss = args.loss
lr = args.lr
training = args.train

if experiment_id is None:
    experiment_id = str(datetime.datetime.now().time()).replace(':', '-').split('.')[0]

HYPERPARAMS = {
    'pretrained': False if (debug and not torch.cuda.is_available()) else True,
    'nb_epochs': 100,
    'scheduler': True,
    'bs': 2 if (debug and not torch.cuda.is_available()) else 24,
    'momentum': 0.9,
    'nesterov': True,
    'weight_decay': 3e-5,
    'early_stopping': False,
    'patience': 10,
    'loss': loss,
    'arcface': {
        's': 30,
        'm': 0.5
    },
    }
HYPERPARAMS['nb_examples'] = HYPERPARAMS['bs'] if debug else None

PATH_DATA = 'data'
PATH_METADATA = os.path.join(PATH_DATA, 'metadata')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    num_workers = 0
else:
    num_workers = 4*torch.cuda.device_count()

if torch.cuda.is_available():
    HYPERPARAMS['bs'] = HYPERPARAMS['bs']*torch.cuda.device_count()
    cudnn.benchmark = True

if lr is None:
    HYPERPARAMS['lr'] = 0.002 * HYPERPARAMS['bs']
else:
    HYPERPARAMS['lr'] = lr

print('Number of workers used:', num_workers, '/', os.cpu_count())
print('Number of GPUs used:', torch.cuda.device_count())

def get_celltype(experiment):
    return experiment.split('-')[0]

nb_classes = 1108
model = TwoSitesNN(pretrained=HYPERPARAMS['pretrained'], nb_classes=nb_classes, loss=loss).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=HYPERPARAMS['lr'], \
    momentum=HYPERPARAMS['momentum'], nesterov=HYPERPARAMS['nesterov'], \
    weight_decay=HYPERPARAMS['weight_decay'])
model = torch.nn.DataParallel(model)

if training:
    print('########## TRAINING ##########')

    df = pd.read_csv(PATH_METADATA+'/train.csv')
    df['celltype'] = df['experiment'].apply(get_celltype)
    df_train, df_val = train_test_split(df, random_state=42)
    if HYPERPARAMS['nb_examples'] is not None:
        df_train = df_train[:HYPERPARAMS['nb_examples']]
        df_val = df_val[:HYPERPARAMS['nb_examples']]
   
    print('Size training dataset: {}'.format(len(df_train)))
    print('Size validation dataset: {}'.format(len(df_val)))

    print('########## TRAINING STEP 1 ##########')
    path_model_step_1 = 'models/best_model_'+experiment_id+'.pth'

    if not os.path.exists(path_model_step_1):
        ds_train = ImagesDS(df=df_train, img_dir=PATH_DATA, mode='train', num_workers=num_workers)
        ds_val = ImagesDS(df=df_val, img_dir=PATH_DATA, mode='train', num_workers=num_workers)

        train(experiment_id, ds_train, ds_val, model, optimizer, HYPERPARAMS, num_workers, device, debug)

    model.load_state_dict(torch.load(path_model_step_1))

    print('\n\n########## TRAINING STEP 2 ##########')

    HYPERPARAMS['pretrained'] = False
    HYPERPARAMS['lr'] = HYPERPARAMS['lr']/10
    HYPERPARAMS['nb_epochs'] = HYPERPARAMS['nb_epochs']//5
    optimizer = torch.optim.SGD(model.parameters(), lr=HYPERPARAMS['lr'], \
        momentum=HYPERPARAMS['momentum'], nesterov=HYPERPARAMS['nesterov'], \
        weight_decay=HYPERPARAMS['weight_decay'])
        
    for celltype in df_train['celltype'].unique():
        df_train_cell = df_train[df_train['celltype']==celltype]
        df_val_cell = df_val[df_val['celltype']==celltype]
        ds_train_cell = ImagesDS(df=df_train_cell, img_dir=PATH_DATA, mode='train', num_workers=num_workers)
        ds_val_cell = ImagesDS(df=df_val_cell, img_dir=PATH_DATA, mode='train', num_workers=num_workers)
        
        model_cell = deepcopy(model)
        model.module.pretrained = False
        experiment_id_cell = experiment_id + '_' + celltype
        path_model_step_2 = 'models/best_model_'+experiment_id_cell+'.pth'
        if not os.path.exists(path_model_step_2):
            print('\nTraining:', celltype)
            train(experiment_id_cell, ds_train_cell, ds_val_cell, model_cell, optimizer, HYPERPARAMS, num_workers, device, debug)

print('\n\n########## TEST ##########')

df_test = pd.read_csv(PATH_METADATA+'/test.csv')
df_test['celltype'] = df_test['experiment'].apply(get_celltype)
print('Size test dataset: {}'.format(len(df_test)))

nb_classes = 1108
model = TwoSitesNN(pretrained=HYPERPARAMS['pretrained'], nb_classes=nb_classes, loss=loss).to(device)
model = torch.nn.DataParallel(model).to(device)
if debug:
    model = DummyClassifier(nb_classes=nb_classes)

# We use the fact that some siRNA are always present on the plates.
plate_groups = np.zeros((1108,4), int)
if debug and (device=='cpu'):
    df = pd.read_csv('data/full_metadata/train.csv')
else:
    df = pd.read_csv('data/metadata/train.csv')
for sirna in range(nb_classes):
    grp = df.loc[df.sirna==sirna,:].plate.value_counts().index.values
    assert len(grp) == 3
    plate_groups[sirna,0:3] = grp
    plate_groups[sirna,3] = 10 - grp.sum()
del df
experiment_types = [3, 1, 0, 0, 0, 0, 2, 2, 3, 0, 0, 3, 1, 0, 0, 0, 2, 3]
# [3 1 0 0 0 0 2 2 3 0 0 3 1 0 0 0 3 3]
# [3 1 0 0 0 0 2 2 3 0 0 3 1 0 0 0 0 3]
idx_experiment = 0

for i, celltype in enumerate(df_test['celltype'].unique()):
    df_test_cell = df_test[df_test['celltype']==celltype]
    experiment_id_cell = experiment_id + '_' + celltype

    for j, experiment in enumerate(df_test_cell['experiment'].unique()):
        df_test_experiment = df_test_cell[df_test_cell['experiment']==experiment]
        ds_test_experiment = ImagesDS(df=df_test_experiment, img_dir=PATH_DATA, mode='test', num_workers=num_workers)

        if not debug:
            model.load_state_dict(torch.load('models/best_model_'+experiment_id+'.pth'))
            model.eval()

        temp = test(experiment_id_cell, df_test_experiment, ds_test_experiment, plate_groups, \
            experiment_types[idx_experiment], model, HYPERPARAMS['bs'], num_workers, device)
        if i==0 and j==0:
            preds = temp
        else:
            preds = np.concatenate([preds, temp], axis=0)
        
        idx_experiment += 1

df_test['sirna'] = preds.astype(int)
df_test.to_csv('submission_' + experiment_id + '.csv', index=False, columns=['id_code','sirna'])
import os
import datetime
import argparse
from copy import deepcopy
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from data_loader import ImagesDS
from models import TwoSitesNN, DummyClassifier

from train import train
from test import test

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='My parser')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--experiment_id')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--train', default=False, action='store_true')

args = parser.parse_args()

debug = args.debug
experiment_id = args.experiment_id
lr = args.lr
training = args.train

if (training and (experiment_id is not None)) or ((not training) and (experiment_id is None)):
    print('Error between "training" and "experiment_id".')
    sys.exit(1)

HYPERPARAMS = {
    'lr' : args.lr,
    'pretrained': True,
    'scheduler': True
}

if debug:
    PATH_DATA = 'data/samples'
    HYPERPARAMS['nb_epochs'] = 1
    HYPERPARAMS['patience'] = 100
    HYPERPARAMS['bs'] = 2
else:
    PATH_DATA = 'data'
    HYPERPARAMS['nb_epochs'] = 1
    HYPERPARAMS['patience'] = 5
    HYPERPARAMS['bs'] = 40

PATH_METADATA = os.path.join(PATH_DATA, 'metadata')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    num_workers = os.cpu_count()
else:
    num_workers = 4*torch.cuda.device_count()

if torch.cuda.is_available():
    HYPERPARAMS['bs'] = HYPERPARAMS['bs']*torch.cuda.device_count()

print('Number of workers used:', num_workers, '/', os.cpu_count())
print('Number of GPUs used:', torch.cuda.device_count())

def get_celltype(experiment):
    return experiment.split('-')[0]

nb_classes = 1108
model = TwoSitesNN(pretrained=HYPERPARAMS['pretrained'], nb_classes=nb_classes)
model = torch.nn.DataParallel(model).to(device)

if training:
    print('########## TRAINING ##########')
    
    experiment_id = str(datetime.datetime.now().time()).replace(':', '-').split('.')[0]

    df = pd.read_csv(PATH_METADATA+'/train.csv')
    df['celltype'] = df['experiment'].apply(get_celltype)
    df_train, df_val = train_test_split(df, test_size = 0.1, random_state=42)
    print('Size training dataset: {}'.format(len(df_train)))
    print('Size validation dataset: {}'.format(len(df_val)))

    print('########## TRAINING STEP 1 ##########')

    ds_train = ImagesDS(df=df_train, img_dir=PATH_DATA, mode='train')
    ds_val = ImagesDS(df=df_val, img_dir=PATH_DATA, mode='train')

    train(experiment_id, ds_train, ds_val, model, HYPERPARAMS, num_workers, device, debug)

    model.load_state_dict(torch.load('models/best_model_'+experiment_id+'.pth'))

    print('\n\n########## TRAINING STEP 2 ##########')

    for celltype in df_train['celltype'].unique():
        print('\nTraining:', celltype)
        df_train_cell = df_train[df_train['celltype']==celltype]
        df_val_cell = df_val[df_val['celltype']==celltype]
        ds_train_cell = ImagesDS(df=df_train_cell, img_dir=PATH_DATA, mode='train')
        ds_val_cell = ImagesDS(df=df_val_cell, img_dir=PATH_DATA, mode='train')
        model_cell = deepcopy(model)
        model.module.pretrained = False
        experiment_id_cell = experiment_id + '_' + celltype
        train(experiment_id_cell, ds_train_cell, ds_val_cell, model_cell, HYPERPARAMS, num_workers, \
            device, debug)

print('\n\n########## TEST ##########')

df_test = pd.read_csv(PATH_METADATA+'/test.csv')
df_test['celltype'] = df_test['experiment'].apply(get_celltype)
print('Size test dataset: {}'.format(len(df_test)))

if debug:
    model = DummyClassifier(nb_classes=nb_classes)

# We use the fact that some siRNA are always present on the plates.
plate_groups = np.zeros((1108,4), int)
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
        ds_test_experiment = ImagesDS(df=df_test_experiment, img_dir=PATH_DATA, mode='test')

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
import os
import datetime
import argparse
from copy import deepcopy
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.backends.cudnn as cudnn

if torch.cuda.is_available():
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from dataloader import ImagesDS
from models import TwoSitesNN, DummyClassifier

from train import train
from test import test

import warnings
warnings.filterwarnings('ignore')

# torch.manual_seed(0)

parser = argparse.ArgumentParser(description='My parser')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--experiment_id')
parser.add_argument('--lr', type=float)
parser.add_argument('--epoch', default=100, type=int)
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
    'pretrained': False if debug else True,
    'nb_epochs': args.epoch,
    'scheduler': True,
    'momentum': 0.9,
    'nesterov': True,
    'weight_decay': 3e-5,
    'early_stopping': False
    }

if debug:
    HYPERPARAMS['nb_examples'] = 10
    HYPERPARAMS['patience'] = 100
    HYPERPARAMS['bs'] = 2
else:
    HYPERPARAMS['nb_examples'] = float('inf')
    HYPERPARAMS['patience'] = 10
    HYPERPARAMS['bs'] = 48
    
PATH_DATA = 'data'
PATH_METADATA = os.path.join(PATH_DATA, 'metadata')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    num_workers = 0
else:
    num_workers = 4*torch.cuda.device_count()

if torch.cuda.is_available():
    HYPERPARAMS['bs'] = HYPERPARAMS['bs']*torch.cuda.device_count()
    # cudnn.benchmark = True

if lr is None:
    HYPERPARAMS['lr'] = 0.002 * HYPERPARAMS['bs']
else:
    HYPERPARAMS['lr'] = lr

print('Number of workers used:', num_workers, '/', os.cpu_count())
print('Number of GPUs used:', torch.cuda.device_count())

def get_celltype(experiment):
    return experiment.split('-')[0]

nb_classes = 1108
model = TwoSitesNN(pretrained=HYPERPARAMS['pretrained'], nb_classes=nb_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=HYPERPARAMS['lr'], \
    momentum=HYPERPARAMS['momentum'], nesterov=HYPERPARAMS['nesterov'], \
    weight_decay=HYPERPARAMS['weight_decay'])
if torch.cuda.is_available():
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
model = torch.nn.DataParallel(model)

if training:
    print('########## TRAINING ##########')
    
    experiment_id = str(datetime.datetime.now().time()).replace(':', '-').split('.')[0]

    df = pd.read_csv(PATH_METADATA+'/train.csv')
    df['celltype'] = df['experiment'].apply(get_celltype)
    df_train, df_val = train_test_split(df, test_size = 0.1, random_state=42)
    df_train = df_train[:HYPERPARAMS['nb_examples']]
    df_val = df_val[:HYPERPARAMS['nb_examples']]
    print('Size training dataset: {}'.format(len(df_train)))
    print('Size validation dataset: {}'.format(len(df_val)))

    print('########## TRAINING STEP 1 ##########')

    ds_train = ImagesDS(df=df_train, img_dir=PATH_DATA, mode='train', num_workers=num_workers)
    ds_val = ImagesDS(df=df_val, img_dir=PATH_DATA, mode='train', num_workers=num_workers)

    train(experiment_id, ds_train, ds_val, model, optimizer, HYPERPARAMS, num_workers, device, debug)

    model.load_state_dict(torch.load('models/best_model_'+experiment_id+'.pth'))

    print('\n\n########## TRAINING STEP 2 ##########')

    if not debug:
        HYPERPARAMS['nb_epochs'] = HYPERPARAMS['nb_epochs']//5
    HYPERPARAMS['lr'] = HYPERPARAMS['lr']/10

    for celltype in df_train['celltype'].unique():
        print('\nTraining:', celltype)
        df_train_cell = df_train[df_train['celltype']==celltype]
        df_val_cell = df_val[df_val['celltype']==celltype]
        ds_train_cell = ImagesDS(df=df_train_cell, img_dir=PATH_DATA, mode='train', num_workers=num_workers)
        ds_val_cell = ImagesDS(df=df_val_cell, img_dir=PATH_DATA, mode='train', num_workers=num_workers)
        model_cell = deepcopy(model)
        model.module.pretrained = False
        experiment_id_cell = experiment_id + '_' + celltype
        train(experiment_id_cell, ds_train_cell, ds_val_cell, model_cell, optimizer, HYPERPARAMS, num_workers, device, debug)

print('\n\n########## TEST ##########')

df_test = pd.read_csv(PATH_METADATA+'/test.csv')
df_test['celltype'] = df_test['experiment'].apply(get_celltype)
print('Size test dataset: {}'.format(len(df_test)))

nb_classes = 1108
model = TwoSitesNN(pretrained=HYPERPARAMS['pretrained'], nb_classes=nb_classes)
model = torch.nn.DataParallel(model).to(device)
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

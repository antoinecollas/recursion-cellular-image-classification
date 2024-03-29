import os
import datetime
import argparse
import pickle
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from sklearn.model_selection import train_test_split
from cell_classifier.dataloader \
    import train_test_split as train_test_split_by_experiment, \
    ImagesDS
from cell_classifier.models import TwoSitesNN, DummyClassifier

from cell_classifier.train import train
from cell_classifier.test import test

import warnings
warnings.filterwarnings('ignore')

# torch.manual_seed(0)

parser = argparse.ArgumentParser(description='My parser')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--experiment_id')
parser.add_argument('--lr', type=float)

args = parser.parse_args()

debug = args.debug
experiment_id = args.experiment_id
lr = args.lr

if experiment_id is None:
    experiment_id = str(datetime.datetime.now().time())
    experiment_id = experiment_id.replace(':', '-').split('.')[0]

local = (debug and not torch.cuda.is_available())
HYPERPARAMS = {
    'train_split_by_experiment': False,
    'pretrained': False if local else True,
    'nb_epochs': 5 if local else 100,
    'scheduler': True,
    'bs': 2 if local else 16,
    'momentum': 0.9,
    'nesterov': True,
    'weight_decay': 3e-5,
    'early_stopping': False,
    'patience': 10,
    }
HYPERPARAMS['nb_examples'] = HYPERPARAMS['bs'] if debug else None

PATH_DATA = 'data'
PATH_METADATA = os.path.join(PATH_DATA, 'metadata')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    num_workers = 0
else:
    num_workers = 4 * torch.cuda.device_count()

if torch.cuda.is_available():
    HYPERPARAMS['bs'] = HYPERPARAMS['bs'] * torch.cuda.device_count()
    cudnn.benchmark = True

if lr is None:
    HYPERPARAMS['lr'] = 0.0005 * HYPERPARAMS['bs']
else:
    HYPERPARAMS['lr'] = lr

print('Number of workers used:', num_workers, '/', os.cpu_count())
print('Number of GPUs used:', torch.cuda.device_count())


def get_celltype(experiment):
    return experiment.split('-')[0]


with open('stats_experiments.pickle', 'rb') as f:
    stats_experiments = pickle.load(f)

nb_classes = 1108
model = TwoSitesNN(pretrained=HYPERPARAMS['pretrained'], nb_classes=nb_classes)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=HYPERPARAMS['lr'],
                            momentum=HYPERPARAMS['momentum'],
                            nesterov=HYPERPARAMS['nesterov'],
                            weight_decay=HYPERPARAMS['weight_decay'])
model = torch.nn.DataParallel(model)

path_model_step_1 = 'models/best_model_'+experiment_id+'.pth'
if not os.path.exists(path_model_step_1):
    print('########## TRAINING ##########')

    df = pd.read_csv(PATH_METADATA+'/train.csv')
    df['celltype'] = df['experiment'].apply(get_celltype)
    if HYPERPARAMS['train_split_by_experiment']:
        df_train, df_val = train_test_split_by_experiment(df, random_state=42)
    else:
        if local:
            stratify = None
        else:
            print('Stratify train/val split by sirna...')
            stratify = df[['sirna']]
        df_train, df_val = train_test_split(df,
                                            test_size=0.1,
                                            random_state=42,
                                            stratify=stratify)

    if HYPERPARAMS['nb_examples'] is not None:
        df_train = df_train[:HYPERPARAMS['nb_examples']]
        df_val = df_val[:HYPERPARAMS['nb_examples']]
    df_controls = pd.read_csv(PATH_METADATA+'/train_controls.csv')

    print('Size training dataset: {}'.format(len(df_train)))
    print('Size validation dataset: {}'.format(len(df_val)))

    ds_train = ImagesDS(df=df_train,
                        df_controls=df_controls,
                        stats_experiments=stats_experiments,
                        img_dir=PATH_DATA,
                        mode='train')
    ds_val = ImagesDS(df=df_val,
                      df_controls=df_controls,
                      stats_experiments=stats_experiments,
                      img_dir=PATH_DATA,
                      mode='val')

    train(experiment_id,
          ds_train,
          ds_val,
          model,
          optimizer,
          HYPERPARAMS,
          num_workers,
          device,
          debug)

if local:
    model = DummyClassifier(nb_classes=nb_classes)
else:
    model.load_state_dict(torch.load(path_model_step_1))
    model.eval()

print('\n\n########## TEST ##########')

df_test = pd.read_csv(PATH_METADATA+'/test.csv')
df_controls = pd.read_csv(PATH_METADATA+'/test_controls.csv')
print('Size test dataset: {}'.format(len(df_test)))

# We use the fact that some siRNA are always present on the plates.
plate_groups = np.zeros((1108, 4), int)
if local:
    df = pd.read_csv('data/full_metadata/train.csv')
else:
    df = pd.read_csv('data/metadata/train.csv')
for sirna in range(nb_classes):
    grp = df.loc[df.sirna == sirna, :].plate.value_counts().index.values
    assert len(grp) == 3
    plate_groups[sirna, 0:3] = grp
    plate_groups[sirna, 3] = 10 - grp.sum()
del df
experiment_types = [3, 1, 0, 0, 0, 0, 2, 2, 3, 0, 0, 3, 1, 0, 0, 0, 2, 3]

idx_experiment = 0
experiments = df_test['experiment'].unique()
if not local:
    assert len(experiment_types) == len(experiments)
for i, experiment in enumerate(tqdm(experiments)):
    df_test_experiment = df_test[df_test['experiment'] == experiment]
    ds_test_experiment = ImagesDS(df=df_test_experiment,
                                  df_controls=df_controls,
                                  stats_experiments=stats_experiments,
                                  img_dir=PATH_DATA,
                                  mode='test',
                                  verbose=False)

    temp = test(df_test_experiment,
                ds_test_experiment,
                plate_groups,
                experiment_types[idx_experiment],
                model,
                HYPERPARAMS['bs'],
                num_workers,
                device)
    if i == 0:
        preds = temp
    else:
        preds = np.concatenate([preds, temp], axis=0)

    idx_experiment += 1

df_test['sirna'] = preds.astype(int)
df_test.to_csv('submission_' + experiment_id + '.csv',
               index=False,
               columns=['id_code', 'sirna'])

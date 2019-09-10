import os, glob, pickle
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd

def compute_mean_std(paths, mean=None, std=None):
    nb_channels = 6
    count = np.zeros(nb_channels)
    sum_x = np.zeros(nb_channels)
    sum_x2 = np.zeros(nb_channels)
    for path in tqdm(paths[:100], desc='Imgs'):
        channel = int(path.split('_')[2][1])-1
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255
        if (mean is not None) and (std is not None):
            im = (im-mean[channel]) / std[channel]
        count[channel] += 1
        sum_x[channel] += np.sum(im)
        sum_x2[channel] += np.sum(im**2)
    count = count*512*512
    mean = sum_x / count
    std = np.sqrt((sum_x2/count) - mean**2)
    return mean, std

FILENAME = 'stats_experiments.pickle'

experiments_train = glob.glob('data/train/*/', recursive=True)
experiments_train = [experiment.split('/')[-2] for experiment in experiments_train]
experiments_test = glob.glob('data/test/*/', recursive=True)
experiments_test = [experiment.split('/')[-2] for experiment in experiments_test]
experiments = experiments_train + experiments_test

stats_experiments = dict()
for experiment in tqdm(experiments, desc='experiments'):
    paths = glob.glob('data/*/'+experiment+'/*/*.jpeg', recursive=True)
    mean, std = compute_mean_std(paths)
    stats_experiments[experiment] = dict()
    stats_experiments[experiment]['mean'] = mean
    stats_experiments[experiment]['std'] = std

with open(FILENAME, 'wb') as f:
    pickle.dump(stats_experiments, f)

print()
print('Verification:')
with open(FILENAME, 'rb') as f:
    stats_experiments = pickle.load(f)

for experiment in experiments:
    paths = glob.glob('data/*/'+experiment+'/*/*.jpeg', recursive=True)
    mean = stats_experiments[experiment]['mean']
    std = stats_experiments[experiment]['std']
    mean, std = compute_mean_std(paths, mean=mean, std=std)
    print('mean=', mean)
    print('std=', std)
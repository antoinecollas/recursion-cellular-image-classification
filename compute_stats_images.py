import os, glob, pickle
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd

def compute_mean_std(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255
    return np.mean(img), np.std(img)

def get_info(path):
    temp = path.split('/')
    experiment = temp[2]
    plate = int(temp[3][-1])
    temp = temp[-1].split('.')[0].split('_')
    well = temp[0]
    site = int(temp[1][-1]) - 1
    channel = int(temp[2][-1]) - 1
    return experiment, plate, well, site, channel

FILENAME = 'stats_images.pickle'

images_train = glob.glob('data/train/*/*/*.jpeg', recursive=True)
images_test = glob.glob('data/test/*/*/*.jpeg', recursive=True)
images = images_train + images_test

stats_images = dict()
N_CHANNELS = 6

for path_image in tqdm(images, desc='images'):
    mean, std = compute_mean_std(path_image)
    experiment, plate, well, site, channel = get_info(path_image)
    if experiment not in stats_images.keys():
        stats_images[experiment] = dict()
    if plate not in stats_images[experiment].keys():
        stats_images[experiment][plate] = dict()
    if well not in stats_images[experiment][plate].keys():
        stats_images[experiment][plate][well] = dict()
    if site not in stats_images[experiment][plate][well].keys():
        stats_images[experiment][plate][well][site] = dict()
        stats_images[experiment][plate][well][site]['mean'] = np.zeros(N_CHANNELS)
        stats_images[experiment][plate][well][site]['std'] = np.zeros(N_CHANNELS)
    stats_images[experiment][plate][well][site]['mean'][channel] = mean
    stats_images[experiment][plate][well][site]['std'][channel] = std

with open(FILENAME, 'wb') as f:
    pickle.dump(stats_images, f)
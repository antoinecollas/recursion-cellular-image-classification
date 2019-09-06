import glob
from tqdm import tqdm
import numpy as np
import cv2

paths = glob.glob('data/**/*.jpeg', recursive=True)
nb_channels = 6

count = np.zeros(nb_channels)
sum_x = np.zeros(nb_channels)
sum_x2 = np.zeros(nb_channels)
for path in tqdm(paths):
    site = int(path.split('_')[1][1])-1
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255
    count[site] += 1
    sum_x[site] += np.sum(im)
    sum_x2[site] += np.sum(im**2)
count = count*512*512
mean = sum_x / count
std = np.sqrt((sum_x2/count) - mean**2)
print('mean=', mean)
print('std=', std)

#verification
count_verification = np.zeros(nb_channels)
sum_x_verification = np.zeros(nb_channels)
sum_x2_verification = np.zeros(nb_channels)
for path in tqdm(paths):
    site = int(path.split('_')[1][1])-1
    im = ((cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255)-mean[site]) / std[site]
    count_verification[site] += 1
    sum_x_verification[site] += np.sum(im)
    sum_x2_verification[site] += np.sum(im**2)
count_verification = count_verification*512*512
mean_verification = sum_x_verification / count_verification
std_verification = np.sqrt((sum_x2_verification/count_verification) - mean_verification**2)
print('mean_verification=', mean_verification)
print('std_verification=', std_verification)
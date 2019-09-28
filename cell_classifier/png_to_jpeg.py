from PIL import Image
import os
import glob
from tqdm import tqdm
import multiprocessing


def update(*a):
    pbar.update()


def convert_png_to_jpeg(path):
    path_jpeg = path.split('.')[0]+'.jpeg'
    im = Image.open(path)
    im = im.convert('L')
    im.save(path_jpeg, quality=95)


paths = glob.glob('data/**/*.png', recursive=True)
pbar = tqdm(total=len(paths))
pool = multiprocessing.Pool(os.cpu_count())
for path in paths:
    pool.apply_async(convert_png_to_jpeg, args=[path], callback=update)
pool.close()
pool.join()

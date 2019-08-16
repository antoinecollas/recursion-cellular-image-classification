import os

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as D
from torch.optim.lr_scheduler import ExponentialLR

from torchvision import models, transforms as T

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import  EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler, GradsHistHandler

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser(description="My parser")
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
debug = args.debug

if debug:
    PATH_DATA = 'data/samples'
    NB_EPOCHS = 5
    PATIENCE = 1000
    BATCH_SIZE = 1
else:
    PATH_DATA = 'data'
    NB_EPOCHS = 100
    PATIENCE = 3
    BATCH_SIZE = 20

if torch.cuda.is_available():
    BATCH_SIZE = BATCH_SIZE * torch.cuda.device_count()

if torch.cuda.device_count() == 8:
    LR = 0.01
elif torch.cuda.device_count() == 4:
    LR = 0.005
else:
    LR = 0.001

PATH_METADATA = os.path.join(PATH_DATA, 'metadata')

print('Number of GPUs available: {}\n'.format(torch.cuda.device_count()))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir, mode='train', site=1, channels=[1,2,3,4,5,6]):
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        
    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return T.ToTensor()(img)

    def _get_img_path(self, index, channel):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir, self.mode, experiment, f'Plate{plate}', f'{well}_s{self.site}_w{channel}.png'])
        
    def __getitem__(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]
        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
        if self.mode == 'train':
            return img, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len

df = pd.read_csv(PATH_METADATA+'/train.csv')
df_train, df_val = train_test_split(df, test_size = 0.1, random_state=42)
df_test = pd.read_csv(PATH_METADATA+'/test.csv')
print('Size training dataset: {}'.format(len(df_train)))
print('Size validation dataset: {}'.format(len(df_val)))
print('Size test dataset: {}\n'.format(len(df_test)))

ds = ImagesDS(df_train, PATH_DATA, mode='train')
ds_val = ImagesDS(df_val, PATH_DATA, mode='train')
ds_test = ImagesDS(df_test, PATH_DATA, mode='test')

classes = 1108
model = models.resnext50_32x4d(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, classes)

# let's make our model work with 6 channels
trained_kernel = model.conv1.weight
new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
model.conv1 = new_conv

model = torch.nn.DataParallel(model)

num_workers = os.cpu_count()
loader = D.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
val_loader = D.DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
tloader = D.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

metrics = {
    'loss': Loss(criterion),
    'accuracy': Accuracy(),
}

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

@trainer.on(Events.EPOCH_STARTED)
def turn_on_layers(engine):
    epoch = engine.state.epoch
    if epoch == 1:
        temp = next(model.named_children())[1]
        for name, child in temp.named_children():
            if name == 'fc':
                print(name + ' is unfrozen')
                for param in child.parameters():
                    param.requires_grad = True
            else:
                print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False

    if epoch == 3:
        print('Turn on all the layers')
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = True

pbar = ProgressBar(bar_format='')
pbar.attach(trainer, output_transform=lambda x: {'loss': x})

val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
handler = EarlyStopping(patience=PATIENCE, score_function=lambda engine: engine.state.metrics['accuracy'], trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, handler)

@trainer.on(Events.EPOCH_COMPLETED)
def compute_and_display_val_metrics(engine):
    epoch = engine.state.epoch
    metrics = val_evaluator.run(val_loader).metrics
    
    if (epoch == 1) or (metrics['accuracy'] > engine.state.best_acc):
        engine.state.best_acc = metrics['accuracy']
        print(f'\nNew best accuracy! Accuracy: {engine.state.best_acc}\nModel saved!')
        if not os.path.exists('models/'):
            os.makedirs('models/')
        torch.save(model.state_dict(), 'models/best_model.pth')

    print('Validation Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f} '
          .format(engine.state.epoch, 
                      metrics['loss'], 
                      metrics['accuracy']))

lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

@trainer.on(Events.EPOCH_COMPLETED)
def update_lr_scheduler(engine):
    lr_scheduler.step()
    lr = float(optimizer.param_groups[0]['lr'])
    print('Learning rate: {}'.format(lr))

tb_logger = TensorboardLogger('board/ResNet18')
tb_logger.attach(trainer, log_handler=OutputHandler(tag='training', output_transform=lambda loss: {'loss': loss}), event_name=Events.ITERATION_COMPLETED)
tb_logger.attach(val_evaluator, log_handler=OutputHandler(tag='validation', metric_names=['accuracy', 'loss'], another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
tb_logger.attach(trainer, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
tb_logger.close()

trainer.run(loader, max_epochs=NB_EPOCHS)

model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()
with torch.no_grad():
    preds = np.empty(0)
    for x, _ in tqdm(tloader):
        x = x.to(device)
        output = model(x)
        idx = output.max(dim=-1)[1].cpu().numpy()
        preds = np.append(preds, idx, axis=0)

submission = pd.read_csv(PATH_METADATA + '/test.csv')
submission['sirna'] = preds.astype(int)
submission.to_csv('submission.csv', index=False, columns=['id_code','sirna'])
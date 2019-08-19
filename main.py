import os
import datetime
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.utils.data as D
import torch.nn as nn

from torchvision import models

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import  EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler, GradsHistHandler

from sklearn.model_selection import train_test_split

from data_loader import ImagesDS
from two_sites_nn import TwoSitesNN

torch.manual_seed(0)

hour = str(datetime.datetime.now().time()).replace(':', '-').split('.')[0]

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
    NB_EPOCHS = 5
    PATIENCE = 1000
    BATCH_SIZE = 3
else:
    PATH_DATA = 'data'
    NB_EPOCHS = 1000
    PATIENCE = 20
    BATCH_SIZE = 40
    
PATH_METADATA = os.path.join(PATH_DATA, 'metadata')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    num_workers = os.cpu_count()
else:
    num_workers = 4*torch.cuda.device_count()

if torch.cuda.is_available():
    BATCH_SIZE = BATCH_SIZE * torch.cuda.device_count()

print('Number of workers used:', num_workers, '/', os.cpu_count())
print('Number of GPUs used:', torch.cuda.device_count())


df = pd.read_csv(PATH_METADATA+'/train.csv')
df_train, df_val = train_test_split(df, test_size = 0.1, random_state=42)
df_test = pd.read_csv(PATH_METADATA+'/test.csv')
print('Size training dataset: {}'.format(len(df_train)))
print('Size validation dataset: {}'.format(len(df_val)))
print('Size test dataset: {}\n'.format(len(df_test)))

ds = ImagesDS(df=df_train, img_dir=PATH_DATA, mode='train')
ds_val = ImagesDS(df=df_val, img_dir=PATH_DATA, mode='train')
ds_test = ImagesDS(df=df_test, img_dir=PATH_DATA, mode='test')

nb_classes = 1108
model = TwoSitesNN(pretrained=pretrain, nb_classes=nb_classes)
model = torch.nn.DataParallel(model)

loader = D.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
val_loader = D.DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
tloader = D.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

metrics = {
    'loss': Loss(criterion),
    'accuracy': Accuracy(),
}

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

if pretrain:
    @trainer.on(Events.EPOCH_STARTED)
    def turn_on_layers(engine):
        epoch = engine.state.epoch
        if epoch == 1:
            temp = next(model.named_children())[1]
            for name, child in temp.named_children():
                if name == 'classifier':
                    print(name + ' is unfrozen')
                    for param in child.parameters():
                        param.requires_grad = True
                else:
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
        torch.save(model.state_dict(), 'models/best_model_'+hour+'.pth')

    print('Validation Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f} '
          .format(engine.state.epoch, 
                      metrics['loss'], 
                      metrics['accuracy']))

if scheduler:
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr_scheduler(engine):
        lr_scheduler.step()

tb_logger = TensorboardLogger('board/' + hour)
tb_logger.attach(trainer, log_handler=OutputHandler(tag='training', output_transform=lambda loss: {'loss': loss}), event_name=Events.ITERATION_COMPLETED)
tb_logger.attach(val_evaluator, log_handler=OutputHandler(tag='validation', metric_names=['accuracy', 'loss'], another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
tb_logger.attach(trainer, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
tb_logger.close()

trainer.run(loader, max_epochs=NB_EPOCHS)

model.load_state_dict(torch.load('models/best_model_'+hour+'.pth'))
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
submission.to_csv('submission_' + hour + '.csv', index=False, columns=['id_code','sirna'])
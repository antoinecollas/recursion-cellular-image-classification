import os

import torch
import torch.nn as nn

from ignite.engine.engine import Engine
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, _prepare_batch
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import  EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler, GradsHistHandler

from loss import ArcFaceLoss

def train(experiment_id, ds_train, ds_val, model, optimizer, hyperparams, num_workers, device, debug):

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=hyperparams['bs'], shuffle=True, \
        num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=hyperparams['bs'], shuffle=True, \
        num_workers=num_workers)

    if hyperparams['loss']=='softmax':
        criterion = nn.CrossEntropyLoss().to(device)
    elif hyperparams['loss']=='arcface':
        criterion = ArcFaceLoss(s=hyperparams['arcface']['s'], m=hyperparams['arcface']['m']).to(device)

    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy(),
    }

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    if hyperparams['pretrained']:
        @trainer.on(Events.EPOCH_STARTED)
        def turn_on_layers(engine):
            epoch = engine.state.epoch
            if epoch == 1:
                print()
                temp = next(model.named_children())[1]
                for name, child in temp.named_children():
                    if (name=='mlp') or (name=='classifier'):
                        print(name + ' is unfrozen')
                        for param in child.parameters():
                            param.requires_grad = True
                    else:
                        for param in child.parameters():
                            param.requires_grad = False

            if epoch == 3:
                print()
                print('Turn on all the layers')
                for name, child in model.named_children():
                    for param in child.parameters():
                        param.requires_grad = True

    pbar = ProgressBar(bar_format='')
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})

    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    
    if hyperparams['early_stopping']:
        handler = EarlyStopping(patience=hyperparams['patience'], score_function=lambda engine: \
            engine.state.metrics['accuracy'], trainer=trainer)
        val_evaluator.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_and_display_val_metrics(engine):
        epoch = engine.state.epoch
        metrics = val_evaluator.run(val_loader).metrics
        
        if (epoch == 1) or (metrics['accuracy'] > engine.state.best_acc):
            engine.state.best_acc = metrics['accuracy']
            print(f'New best accuracy! Accuracy: {engine.state.best_acc}\nModel saved!')
            if not os.path.exists('models/'):
                os.makedirs('models/')
            torch.save(model.state_dict(), 'models/best_model_'+experiment_id+'.pth')

        print('Validation Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f} '
            .format(engine.state.epoch, 
                        metrics['loss'], 
                        metrics['accuracy']))

    if hyperparams['scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, hyperparams['nb_epochs'], \
            eta_min=hyperparams['lr']/100, last_epoch=-1)
        @trainer.on(Events.EPOCH_COMPLETED)
        def update_lr_scheduler(engine):
            lr_scheduler.step()

    tb_logger = TensorboardLogger('board/'+experiment_id)
    tb_logger.attach(trainer, log_handler=OutputHandler(tag='training', \
        output_transform=lambda loss: {'loss': loss}), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(val_evaluator, log_handler=OutputHandler(tag='validation', \
        metric_names=['accuracy', 'loss'], another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), \
        event_name=Events.ITERATION_STARTED)
    tb_logger.attach(trainer, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
    tb_logger.close()

    trainer.run(train_loader, max_epochs=hyperparams['nb_epochs'])
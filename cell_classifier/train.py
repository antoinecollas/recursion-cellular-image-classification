import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from ignite.engine import Events, create_supervised_evaluator, \
                          create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, \
                                                       OutputHandler, \
                                                       OptimizerParamsHandler,\
                                                       GradsHistHandler


def train(experiment_id,
          ds_train,
          ds_val,
          model,
          optimizer,
          hyperparams,
          num_workers,
          device,
          debug):

    train_loader = torch.utils.data.DataLoader(ds_train,
                                               batch_size=hyperparams['bs'],
                                               shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(ds_val,
                                             batch_size=hyperparams['bs'],
                                             shuffle=True,
                                             num_workers=num_workers)

    criterion = nn.CrossEntropyLoss().to(device)

    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy(),
    }

    trainer = create_supervised_trainer(model, optimizer, criterion, device)

    if hyperparams['pretrained']:
        @trainer.on(Events.EPOCH_STARTED)
        def turn_on_layers(engine):
            epoch = engine.state.epoch
            if epoch == 1:
                print()
                temp = next(model.named_children())[1]
                for name, child in temp.named_children():
                    if (name == 'mlp') or (name == 'classifier'):
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

    val_evaluator = create_supervised_evaluator(model, metrics, device)

    if hyperparams['early_stopping']:
        def score_function(engine):
            return engine.state.metrics['accuracy']
        handler = EarlyStopping(patience=hyperparams['patience'],
                                score_function=score_function,
                                trainer=trainer)
        val_evaluator.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.STARTED)
    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_and_display_val_metrics(engine):
        epoch = engine.state.epoch
        metrics = val_evaluator.run(val_loader).metrics

        if (epoch == 0) or (metrics['accuracy'] > engine.state.best_acc):
            engine.state.best_acc = metrics['accuracy']
            print('New best accuracy! Accuracy: ' +
                  str(engine.state.best_acc) +
                  '\nModel saved!')
            if not os.path.exists('models/'):
                os.makedirs('models/')
            path = 'models/best_model_'+experiment_id+'.pth'
            torch.save(model.state_dict(), path)

        print('Validation Results - Epoch: {} \
              Average Loss: {:.4f} | Accuracy: {:.4f}'
              .format(engine.state.epoch,
                      metrics['loss'],
                      metrics['accuracy']))

    if hyperparams['scheduler']:
        lr_scheduler = CosineAnnealingLR(optimizer,
                                         hyperparams['nb_epochs'],
                                         eta_min=hyperparams['lr']/100,
                                         last_epoch=-1)

        @trainer.on(Events.EPOCH_COMPLETED)
        def update_lr_scheduler(engine):
            lr_scheduler.step()

    tb_logger = TensorboardLogger('board/'+experiment_id)

    def output_transform(loss):
        return {'loss': loss}

    log_handler = OutputHandler('training', output_transform)
    tb_logger.attach(trainer,
                     log_handler,
                     event_name=Events.ITERATION_COMPLETED)
    log_handler = OutputHandler(tag='validation',
                                metric_names=['accuracy', 'loss'],
                                another_engine=trainer)
    tb_logger.attach(val_evaluator,
                     log_handler,
                     event_name=Events.STARTED)
    tb_logger.attach(val_evaluator,
                     log_handler,
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=OptimizerParamsHandler(optimizer),
                     event_name=Events.ITERATION_STARTED)
    tb_logger.attach(trainer,
                     log_handler=GradsHistHandler(model),
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.close()

    trainer.run(train_loader, max_epochs=hyperparams['nb_epochs'])

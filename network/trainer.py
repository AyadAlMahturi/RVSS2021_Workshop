import datetime
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import cmd_printer

# torch.manual_seed(1)


class Trainer:
    def __init__(self, args):
        print(f'\n=> The device is using {torch.cuda.device_count()} GPU(s).')
        self.args = args
        self.lowest_loss = 10
        self.loss_reduction = 0
        self.last_epoch = -1
        self.current_epoch = None
        self.gpu_count = torch.cuda.device_count()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        #
        if self.args.output_folder == '':
            raise Exception('Output Destination cannot be empty !!!')
        weights_dir = os.path.join(args.target_bucket, 'weights')
        self.output_path = os.path.join(weights_dir, args.output_folder)
        os.makedirs(self.output_path, exist_ok=True)

    def fit(self, model, train_loader, eval_loader):
        model = model.to(self.device)
        optimiser = model.get_optimiser()
        lr_scheduler = model.get_lr_scheduler(optimiser)
        model, optimiser, lr_scheduler = self.load_ckpt(model, optimiser,
                                                        lr_scheduler)
        if self.last_epoch == -1:
            self.init_log(model)
        for epoch_idx in range(self.last_epoch + 1, self.args.epochs):
            self.current_epoch = epoch_idx
            clock = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.log(cmd_printer.divider(
                text=f'Epoch {epoch_idx} | {clock}', show=False))
            self.log(f"\n=> Current Lr: {optimiser.param_groups[0]['lr']}")
            model = model.train()
            loss_buff = []
            n_batches = len(train_loader)
            with tqdm(train_loader, unit=' batches',
                      bar_format='{l_bar}{bar:10}{r_bar}') as t_epoch:
                for batch_idx, batch in enumerate(t_epoch):
                    t_epoch.set_description(
                        f'Epoch {epoch_idx}/{self.args.epochs - 1}')
                    optimiser.zero_grad()
                    # Forwards
                    batch = [x.to(self.device) for x in batch]
                    loss = model.step(batch)
                    loss.backward()
                    optimiser.step()
                    loss_buff.append(loss.item())
                    t_epoch.set_postfix(loss=f'{loss.item():.4f}')
                    if batch_idx % self.args.log_freq == 0:
                        self.log(
                            f'\n[{batch_idx}/{n_batches}]: {loss.item():.4f}')
                avg_train_loss = np.mean(loss_buff)
                loss_eval = self.evaluate(model, eval_loader)
                #
                if lr_scheduler is not None:
                    lr_scheduler.step()
                self.loss_reduction = self.lowest_loss - loss_eval
                if self.loss_reduction > 0:
                    self.lowest_loss = loss_eval
                # output to log
                self.log(
                    f'\n=> Training Loss: {avg_train_loss:.4f}, ' + \
                    f'Evaluation Loss {loss_eval:.4f}')
                self.log('\n')
                # output to terminal
                print(
                    f'=> Training Loss: {avg_train_loss:.4f} , ' + \
                    f'Evaluation Loss {loss_eval:.4f}')
                self.save_ckpt(model, optimiser, lr_scheduler)

    def evaluate(self, model, eval_loader):
        model = model.eval()
        with torch.no_grad():
            loss_buff = []
            with tqdm(
                    eval_loader, unit=' batches',
                    bar_format='{desc}{percentage:3.0f}%{postfix}') as i_epoch:
                i_epoch.set_description('Evaluation')
                for batch in i_epoch:
                    # forward propagation
                    batch = [x.to(self.device) for x in batch]
                    loss_eval_temp = model.step(batch)
                    loss_buff.append(loss_eval_temp.item())
                    i_epoch.set_postfix(loss=f'{loss_eval_temp.item():.4f}')
        loss_eval = np.mean(loss_buff)
        return loss_eval

    # tool box
    def load_ckpt(self, model, optimiser=None, lr_scheduler=None):
        ckpt_suffix = '.best.pth.tar' if self.args.load_best else '.pth.tar'
        ckpt_name = f'{self.args.output_folder}{ckpt_suffix}'
        ckpt_path = os.path.join(self.output_path, ckpt_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path,
                              map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt['weights'])
            if optimiser is not None:
                optimiser.load_state_dict(ckpt['optimiser'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            self.last_epoch = ckpt['last_epoch']
            self.lowest_loss = ckpt['lowest_loss']
            print(f'=> Loaded from {ckpt_name}, Epoch {self.last_epoch}\n')
        if self.gpu_count > 1:
            print(f'Training with {self.gpu_count} GPUs')
            model = nn.DataParallel(model)
        return model, optimiser, lr_scheduler

    def save_ckpt(self, model, optimiser, lr_scheduler=None):
        try:
            weights = model.state_dict()
        except AttributeError:
            weights = model.module.state_dict()
        ckpt = {'last_epoch': self.current_epoch,
                'weights': weights,
                'optimiser': optimiser.state_dict(),
                'lowest_loss': self.lowest_loss
                }
        if optimiser is not None:
            ckpt['lr_scheduler'] = lr_scheduler.state_dict()
        ckpt_name = f'{self.args.output_folder}.pth.tar'
        ckpt_path = os.path.join(self.output_path, ckpt_name)
        torch.save(ckpt, ckpt_path)  # save checkpoint
        if self.loss_reduction > 0:
            best_ckpt_name = f'{self.args.output_folder}.best.pth.tar'
            best_file_path = os.path.join(self.output_path, best_ckpt_name)
            shutil.copyfile(ckpt_path, best_file_path)
            if self.current_epoch > 0:
                print(
                    f'=> Best Model Updated, {self.loss_reduction:.3f} ' + \
                    'Eval Loss Reduction\n')
            else:
                print('\n')
        else:
            print('=> Model Saved\n')

    def log(self, item):
        with open(os.path.join(self.output_path, 'log.txt'), 'a') as log_file:
            log_file.write(item)

    def init_log(self, model):
        with open(os.path.join(self.output_path, 'log.txt'), 'a') as _f:
            print('Net Architecture:', file=_f)
            print(model, file=_f)
            _f.write(f'Loss Function: {model.criterion.__class__.__name__}\n')
            _f.write(cmd_printer.divider(show=False))
            _f.write(cmd_printer.divider(text='Hyper-parameters', show=False))
            for arg in vars(self.args):
                _f.write(f'\n{arg}: {getattr(self.args, arg)}')
            _f.write(cmd_printer.divider(show=False))

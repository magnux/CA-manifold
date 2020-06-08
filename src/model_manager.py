import torch
from torch.nn.utils import clip_grad_norm_
from src.log_manager import LogManager
from src.checkpoint_manager import CheckpointManager
from src.config import build_network, build_optimizer, build_lr_scheduler
from src.utils.model_utils import toggle_grad, count_parameters, make_grad_safe
from os import path
from contextlib import contextmanager
from glob import glob


class ModelManager(object):
    def __init__(self, model_name, networks_dict, config, logging=True):
        self.model_name = model_name
        self.networks_dict = networks_dict
        self.config = config
        self.logging = logging

        for net_name in self.networks_dict.keys():

            self.networks_dict[net_name]['net'] = build_network(self.config, self.networks_dict[net_name]['class'],
                                                                self.networks_dict[net_name]['sub_class'])
            if torch.cuda.is_available():
                self.networks_dict[net_name]['net'].to("cuda:0")
                if torch.cuda.device_count() > 1:
                    self.networks_dict[net_name]['net'] = torch.nn.DataParallel(self.networks_dict[net_name]['net'])

            self.networks_dict[net_name]['optimizer'] = build_optimizer(self.networks_dict[net_name]['net'], self.config)

        self.checkpoint_manager = CheckpointManager(self.config['training']['out_dir'])
        self.checkpoint_manager.register_modules(**{net_name: self.networks_dict[net_name]['net']
                                                    for net_name in self.networks_dict.keys()})
        self.checkpoint_manager.register_modules(**{'%s_optimizer' % net_name: self.networks_dict[net_name]['optimizer']
                                                    for net_name in self.networks_dict.keys()})
        self.start_epoch = self.checkpoint_manager.load_last(self.model_name)

        if self.config['training']['lr_anneal_every'] > 0:
            for net_name in self.networks_dict.keys():
                self.networks_dict[net_name]['lr_scheduler'] = build_lr_scheduler(self.networks_dict[net_name]['optimizer'],
                                                                                  self.config, self.start_epoch)

        if self.logging:
            self.log_manager = LogManager(log_dir=path.join(self.config['training']['out_dir'], '%s_logs' % self.model_name),
                                          img_dir=path.join(self.config['training']['out_dir'], '%s_imgs' % self.model_name),
                                          txt_dir=path.join(self.config['training']['out_dir'], '%s_txts' % self.model_name),
                                          monitoring=self.config['training']['monitoring'],
                                          monitoring_dir=path.join(self.config['training']['out_dir'], '%s_monitoring' % self.model_name))

            if self.start_epoch != 0:
                self.log_manager.load_stats('%s_stats.p' % self.model_name)

        self.epoch = self.start_epoch
        self.lr = self.config['training']['lr']

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def print(self):
        for net_name in self.networks_dict.keys():
            print('# %s parameters:' % net_name, count_parameters(self.networks_dict[net_name]['net']))
            # print(self.networks_dict[net_name]['net'])

    def get_network(self, net_name):
        return self.networks_dict[net_name]['net']

    def set_lr(self, lr):
        self.lr = lr
        for net_name in self.networks_dict.keys():
            for param_group in self.networks_dict[net_name]['optimizer'].param_groups:
                param_group['lr'] = self.lr

    def set_n_calls(self, net_name, n_calls):
        if isinstance(self.networks_dict[net_name]['net'], torch.nn.DataParallel):
            self.networks_dict[net_name]['net'].module.n_calls = n_calls
        else:
            self.networks_dict[net_name]['net'].n_calls = n_calls

    def on_epoch_start(self, epoch):
        self.epoch = epoch

        if self.config['training']['lr_anneal_every'] > 0:
            for net_name in self.networks_dict.keys():
                self.lr = self.networks_dict[net_name]['lr_scheduler'].get_lr()[0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_epoch_end(self):
        if self.config['training']['save_every'] > 0 and ((self.epoch + 1) % self.config['training']['save_every']) == 0:
            self.checkpoint_manager.save(self.epoch + 1, '%s_%06d.pt' % (self.model_name, self.epoch + 1))
            if self.logging:
                self.log_manager.save_stats('%s_stats.p' % self.model_name)
                self.log_manager.flush()

        if self.config['training']['lr_anneal_every'] > 0:
            for net_name in self.networks_dict.keys():
                self.networks_dict[net_name]['lr_scheduler'].step(self.epoch)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def on_epoch(self, epoch):
        self.on_epoch_start(epoch)
        yield None
        self.on_epoch_end()

    def on_batch_start(self):
        pass

    def on_batch_end(self):
        pass

    @contextmanager
    def on_batch(self):
        self.on_batch_start()
        yield None
        self.on_batch_end()

    def on_step_start(self, nets_to_train):
        for net_name in self.networks_dict.keys():
            if net_name in nets_to_train:
                toggle_grad(self.networks_dict[net_name]['net'], True)
                self.networks_dict[net_name]['optimizer'].zero_grad()
            else:
                toggle_grad(self.networks_dict[net_name]['net'], False)

    def on_step_end(self, nets_to_train):
        for net_name in self.networks_dict.keys():
            if net_name in nets_to_train:
                make_grad_safe(self.networks_dict[net_name]['net'])
                clip_grad_norm_(self.networks_dict[net_name]['net'].parameters(), 1., torch._six.inf)
                self.networks_dict[net_name]['optimizer'].step()
                toggle_grad(self.networks_dict[net_name]['net'], False)

    @contextmanager
    def on_step(self, nets_to_train):
        self.on_step_start(nets_to_train)
        yield nets_to_train
        self.on_step_end(nets_to_train)

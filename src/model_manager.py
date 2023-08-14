import torch
from torch.nn.utils import clip_grad_norm_
from src.log_manager import LogManager
from src.checkpoint_manager import CheckpointManager
from src.config import build_network, build_optimizer, build_lr_scheduler
from src.utils.model_utils import toggle_grad, count_parameters, make_grad_safe, update_network_average
from src.optimizers.lr_scheduler import StepLRm
from os import path
from contextlib import contextmanager
import numpy as np
import copy

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


class ModelManager(object):
    def __init__(self, model_name, networks_dict, config, logging=True, to_avg=[], clip_grad=False):
        self.model_name = model_name
        self.networks_dict = networks_dict
        self.config = config
        self.logging = logging
        self.to_avg = to_avg
        self.clip_grad = clip_grad
        self.fp16 = config['training']['fp16'] if 'fp16' in config['training'] else False
        assert not self.fp16 or self.fp16 and APEX_AVAILABLE, 'Apex is not available for fp16 training'

        # config_sgd = copy.deepcopy(self.config)
        # config_sgd['training']['optimizer'] = 'sgd'

        for net_name in self.networks_dict.keys():

            self.networks_dict[net_name]['net'] = build_network(self.config, self.networks_dict[net_name]['class'],
                                                                self.networks_dict[net_name]['sub_class'])
            if torch.cuda.is_available():
                self.networks_dict[net_name]['net'].to("cuda:0")
                if torch.cuda.device_count() > 1:
                    self.networks_dict[net_name]['dp_net'] = torch.nn.DataParallel(self.networks_dict[net_name]['net'])

            if net_name in self.to_avg:
                self.networks_dict[net_name]['avg'] = build_network(self.config, self.networks_dict[net_name]['class'],
                                                                    self.networks_dict[net_name]['sub_class'])
                if torch.cuda.is_available():
                    self.networks_dict[net_name]['avg'].to("cuda:0")
                toggle_grad(self.networks_dict[net_name]['avg'], False)

            if len(list(self.networks_dict[net_name]['net'].parameters())) == 0:
                print('Warning, network without trainable parameters: ', self.networks_dict[net_name]['net'])
            else:
                self.networks_dict[net_name]['optimizer'] = build_optimizer(self.networks_dict[net_name]['net'], self.config)

            self.networks_dict[net_name]['grad_norm'] = 1.

            if self.fp16:
                self.networks_dict[net_name]['net'], self.networks_dict[net_name]['optimizer'] = amp.initialize(
                    self.networks_dict[net_name]['net'], self.networks_dict[net_name]['optimizer'], opt_level='O2', loss_scale=1., verbosity=0)

        self.checkpoint_manager = CheckpointManager(self.config['training']['out_dir'])
        self.checkpoint_manager.register_modules(**{net_name: self.networks_dict[net_name]['net']
                                                    for net_name in self.networks_dict.keys()})
        if len(self.to_avg) > 0:
            self.checkpoint_manager.register_modules(**{'%s_avg' % net_name: self.networks_dict[net_name]['avg']
                                                        for net_name in self.to_avg})
        self.checkpoint_manager.register_modules(**{'%s_optimizer' % net_name: self.networks_dict[net_name]['optimizer']
                                                    for net_name in self.networks_dict.keys() if 'optimizer' in self.networks_dict[net_name]})
        self.it = self.checkpoint_manager.load_last(self.model_name)
        self.epoch = self.it // config['training']['batches_per_epoch']

        # for net_name in self.to_avg:
        #     update_network_average(self.networks_dict[net_name]['avg'], self.networks_dict[net_name]['net'], 0.0)

        if self.config['training']['lr_anneal_every'] > 0:
            for net_name in self.networks_dict.keys():
                if 'optimizer' in self.networks_dict[net_name]:
                    self.networks_dict[net_name]['lr_scheduler'] = build_lr_scheduler(self.networks_dict[net_name]['optimizer'],
                                                                                      self.config, self.epoch)

        if self.logging:
            self.log_manager = LogManager(log_dir=path.join(self.config['training']['out_dir'], '%s_logs' % self.model_name),
                                          img_dir=path.join(self.config['training']['out_dir'], '%s_imgs' % self.model_name),
                                          txt_dir=path.join(self.config['training']['out_dir'], '%s_txts' % self.model_name),
                                          monitoring=self.config['training']['monitoring'],
                                          monitoring_dir=path.join(self.config['training']['out_dir'], '%s_monitoring' % self.model_name))

            if self.it != 0:
                self.log_manager.load_stats('%s_stats.p' % self.model_name)

        self.lr = self.config['training']['lr']
        self.momentum = None
        self.err = np.inf
        self.best_err = np.inf

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save(self, save_filename=None):
        if save_filename is None:
            self.checkpoint_manager.prune_saves(self.model_name)
            self.checkpoint_manager.save(self.it, '%s_%010d.pt' % (self.model_name, self.it))
            if self.logging:
                self.log_manager.save_stats('%s_stats.p' % self.model_name)
                self.log_manager.flush()
        else:
            self.checkpoint_manager.save(self.it, self.model_name + '_' + save_filename + '.pt')
            if self.logging:
                self.log_manager.save_stats(self.model_name + '_' + save_filename + '_stats.p')
                self.log_manager.flush()

    def print(self):
        for net_name in self.networks_dict.keys():
            print('# %s parameters:' % net_name, count_parameters(self.networks_dict[net_name]['net']))
            # print(self.networks_dict[net_name]['net'])

    def get_network(self, net_name, avg=False):
        if avg:
            return self.networks_dict[net_name]['avg']
        elif torch.cuda.device_count() > 1:
            return self.networks_dict[net_name]['dp_net']
        else:
            return self.networks_dict[net_name]['net']

    def set_lr(self, lr):
        self.lr = lr
        for net_name in self.networks_dict.keys():
            for param_group in self.networks_dict[net_name]['optimizer'].param_groups:
                param_group['lr'] = self.lr

    def adjust_betas(self, nets_to_adjust, loss_sign, sign_target_beta1, sign_target_beta2, lr=1e-2):
        d_beta1 = (sign_target_beta1 - abs(loss_sign)) / sign_target_beta1
        d_beta2 = (sign_target_beta2 - abs(loss_sign)) / sign_target_beta2

        for net_name in self.networks_dict.keys():
            if net_name in nets_to_adjust:
                for param_group in self.networks_dict[net_name]['optimizer'].param_groups:
                    beta1, beta2 = param_group['betas']
                    beta1 += lr * d_beta1
                    beta1 = np.clip(beta1, 0., 0.99)
                    beta2 += lr * d_beta2
                    beta2 = np.clip(beta2, 0., 0.999)
                    param_group['betas'] = (beta1, beta2)
        return beta1, beta2

    def set_grad_norm(self, net_name, grad_norm):
        self.networks_dict[net_name]['grad_norm'] = grad_norm

    def get_n_calls(self, net_name):
        return self.networks_dict[net_name]['net'].n_calls

    def set_n_calls(self, net_name, n_calls, avg=False):
        if avg:
            self.networks_dict[net_name]['avg'].n_calls = n_calls
        else:
            self.networks_dict[net_name]['net'].n_calls = n_calls

    def set_lr_mul(self, net_name, lr_mul):
        self.networks_dict[net_name]['net'].lr_mul = lr_mul

    def set_err(self, err):
        self.err = err

    def on_epoch_start(self):
        if self.config['training']['lr_anneal_every'] > 0 and not (
                self.config['training']['lr_anneal_every'] == 1 and self.config['training']['lr_anneal'] == 1.0):
            for net_name in self.networks_dict.keys():
                if 'optimizer' in self.networks_dict[net_name]:
                    self.lr = self.networks_dict[net_name]['lr_scheduler'].get_last_lr()[0]
                    if isinstance(self.networks_dict[net_name]['lr_scheduler'], StepLRm):
                        self.momentum = self.networks_dict[net_name]['lr_scheduler'].get_last_momentum()[0][0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_epoch_end(self):
        self.epoch += 1
        if self.config['training']['lr_anneal_every'] > 0 and not (
                self.config['training']['lr_anneal_every'] == 1 and self.config['training']['lr_anneal'] == 1.0):
            for net_name in self.networks_dict.keys():
                if 'optimizer' in self.networks_dict[net_name]:
                    self.networks_dict[net_name]['lr_scheduler'].step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def on_epoch(self):
        self.on_epoch_start()
        yield None
        self.on_epoch_end()

    def on_batch_start(self):
        if self.config['training']['lr_anneal_every'] == 1 and self.config['training']['lr_anneal'] == 1.0:
            for net_name in self.networks_dict.keys():
                if 'optimizer' in self.networks_dict[net_name]:
                    self.lr = self.networks_dict[net_name]['lr_scheduler'].get_last_lr()[0]

    def on_batch_end(self):
        self.it += 1
        #TODO: Check all networks_dict are trained
        if self.config['training']['save_every'] > 0 and (self.it % self.config['training']['save_every']) == 0:
            for net_name in self.to_avg:
                # if self.it < 1024:
                #     update_network_average(self.networks_dict[net_name]['avg'], self.networks_dict[net_name]['net'], 0.0)
                # else:
                update_network_average(self.networks_dict[net_name]['avg'], self.networks_dict[net_name]['net'], 0.9)

            self.save()

            if self.err < self.best_err:
                self.best_err = self.err
                self.save("best")

        if self.config['training']['lr_anneal_every'] == 1 and self.config['training']['lr_anneal'] == 1.0:
            for net_name in self.networks_dict.keys():
                if 'optimizer' in self.networks_dict[net_name]:
                    self.networks_dict[net_name]['lr_scheduler'].step()

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
        #TODO: Check all nets_to_train are trained
        for net_name in self.networks_dict.keys():
            if net_name in nets_to_train:
                if self.clip_grad:
                    try:
                        if self.fp16:
                            make_grad_safe(amp.master_params(self.networks_dict[net_name]['optimizer']), -(2 ** 15), 2 ** 15)
                            clip_grad_norm_(amp.master_params(self.networks_dict[net_name]['optimizer']), self.networks_dict[net_name]['grad_norm'], torch._six.inf)
                        else:
                            # make_grad_safe(self.networks_dict[net_name]['net'].parameters(), -(2 ** 31), 2 ** 31)
                            clip_grad_norm_(self.networks_dict[net_name]['net'].parameters(), self.networks_dict[net_name]['grad_norm'], torch._six.inf)
                    except ValueError:
                        print('ValueError. Skipping grad clipping.')
                self.networks_dict[net_name]['optimizer'].step()
                toggle_grad(self.networks_dict[net_name]['net'], False)

    @contextmanager
    def on_step(self, nets_to_train):
        self.on_step_start(nets_to_train)
        yield nets_to_train
        self.on_step_end(nets_to_train)

    def loss_backward(self, loss, nets_to_train, **kwargs):
        if self.fp16:
            optimizers = []
            for net_name in self.networks_dict.keys():
                if net_name in nets_to_train:
                    optimizers.append(self.networks_dict[net_name]['optimizer'])
            with amp.scale_loss(loss, optimizers) as scaled_loss:
                scaled_loss.backward(**kwargs)
        else:
            loss.backward(**kwargs)

    def log_images(self, images, tag):
        self.log_manager.add_imgs(images, tag, self.it)

    def log_scalar(self, category, k, v):
        self.log_manager.add_scalar(category, k, v, self.it)

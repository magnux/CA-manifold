import torch
from torch.nn.utils import clip_grad_norm_
from src.log_manager import LogManager
from src.checkpoint_manager import CheckpointManager
from src.config import build_network, build_optimizer, build_lr_scheduler
from src.utils.model_utils import toggle_grad, count_parameters, make_grad_safe, update_network_average
from src.optimizers.lr_scheduler import StepLRm
from os import path
from contextlib import contextmanager

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


class ModelManager(object):
    def __init__(self, model_name, networks_dict, config, logging=True, to_avg=[]):
        self.model_name = model_name
        self.networks_dict = networks_dict
        self.config = config
        self.logging = logging
        self.to_avg = to_avg
        self.fp16 = config['training']['fp16'] if 'fp16' in config['training'] else False
        assert not self.fp16 or self.fp16 and APEX_AVAILABLE, 'Apex is not available for fp16 training'

        for net_name in self.networks_dict.keys():

            self.networks_dict[net_name]['net'] = build_network(self.config, self.networks_dict[net_name]['class'],
                                                                self.networks_dict[net_name]['sub_class'])
            if torch.cuda.is_available():
                self.networks_dict[net_name]['net'].to("cuda:0")
                if torch.cuda.device_count() > 1:
                    self.networks_dict[net_name]['net'] = torch.nn.DataParallel(self.networks_dict[net_name]['net'])

            if net_name in self.to_avg:
                self.networks_dict[net_name]['avg'] = build_network(self.config, self.networks_dict[net_name]['class'],
                                                                    self.networks_dict[net_name]['sub_class'])
                if torch.cuda.is_available():
                    self.networks_dict[net_name]['avg'].to("cuda:0")
                    if torch.cuda.device_count() > 1:
                        self.networks_dict[net_name]['avg'] = torch.nn.DataParallel(self.networks_dict[net_name]['avg'])
                toggle_grad(self.networks_dict[net_name]['avg'], False)

            if len(list(self.networks_dict[net_name]['net'].parameters())) == 0:
                print('Warning, network without trainable parameters: ', self.networks_dict[net_name]['net'])
            else:
                self.networks_dict[net_name]['optimizer'] = build_optimizer(self.networks_dict[net_name]['net'], self.config)

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
        self.start_epoch = self.checkpoint_manager.load_last(self.model_name)

        if self.config['training']['lr_anneal_every'] > 0:
            for net_name in self.networks_dict.keys():
                if 'optimizer' in self.networks_dict[net_name]:
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
        self.momentum = None
        self.it = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def print(self):
        for net_name in self.networks_dict.keys():
            print('# %s parameters:' % net_name, count_parameters(self.networks_dict[net_name]['net']))
            # print(self.networks_dict[net_name]['net'])

    def get_network(self, net_name):
        return self.networks_dict[net_name]['net']

    def get_network_avg(self, net_name):
        return self.networks_dict[net_name]['avg']

    def set_lr(self, lr):
        self.lr = lr
        for net_name in self.networks_dict.keys():
            for param_group in self.networks_dict[net_name]['optimizer'].param_groups:
                param_group['lr'] = self.lr

    def get_n_calls(self, net_name):
        if isinstance(self.networks_dict[net_name]['net'], torch.nn.DataParallel):
            return self.networks_dict[net_name]['net'].module.n_calls
        else:
            return self.networks_dict[net_name]['net'].n_calls

    def set_n_calls(self, net_name, n_calls):
        if isinstance(self.networks_dict[net_name]['net'], torch.nn.DataParallel):
            self.networks_dict[net_name]['net'].module.n_calls = n_calls
        else:
            self.networks_dict[net_name]['net'].n_calls = n_calls

    def set_lr_mul(self, net_name, lr_mul):
        if isinstance(self.networks_dict[net_name]['net'], torch.nn.DataParallel):
            self.networks_dict[net_name]['net'].module.lr_mul = lr_mul
        else:
            self.networks_dict[net_name]['net'].lr_mul = lr_mul

    def on_epoch_start(self, epoch):
        self.epoch = epoch

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
        if self.config['training']['save_every'] > 0 and ((self.epoch + 1) % self.config['training']['save_every']) == 0:
            self.checkpoint_manager.save(self.epoch + 1, '%s_%06d.pt' % (self.model_name, self.epoch + 1))
            self.checkpoint_manager.prune_saves(self.model_name)
            if self.logging:
                self.log_manager.save_stats('%s_stats.p' % self.model_name)
                self.log_manager.flush()

        if len(self.to_avg) > 0:
            for net_name in self.to_avg:
                if self.epoch < 2:
                    update_network_average(self.networks_dict[net_name]['avg'], self.networks_dict[net_name]['net'], 0.0)
                else:
                    update_network_average(self.networks_dict[net_name]['avg'], self.networks_dict[net_name]['net'], 0.999)

        if self.config['training']['lr_anneal_every'] > 0 and not (
                self.config['training']['lr_anneal_every'] == 1 and self.config['training']['lr_anneal'] == 1.0):
            for net_name in self.networks_dict.keys():
                if 'optimizer' in self.networks_dict[net_name]:
                    self.networks_dict[net_name]['lr_scheduler'].step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def on_epoch(self, epoch):
        self.on_epoch_start(epoch)
        yield None
        self.on_epoch_end()

    def on_batch_start(self):
        if self.config['training']['lr_anneal_every'] == 1 and self.config['training']['lr_anneal'] == 1.0:
            for net_name in self.networks_dict.keys():
                if 'optimizer' in self.networks_dict[net_name]:
                    self.lr = self.networks_dict[net_name]['lr_scheduler'].get_last_lr()[0]

    def on_batch_end(self):
        #TODO: Check all networks_dict are trained
        if len(self.to_avg) > 0 and self.epoch >= 2 and self.it % 10 == 0:
            for net_name in self.to_avg:
                update_network_average(self.networks_dict[net_name]['avg'], self.networks_dict[net_name]['net'], 0.999)
        self.it += 1
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
                try:
                    if self.fp16:
                        make_grad_safe(amp.master_params(self.networks_dict[net_name]['optimizer']), -(2 ** 15), 2 ** 15)
                        clip_grad_norm_(amp.master_params(self.networks_dict[net_name]['optimizer']), 1., torch._six.inf)
                    else:
                        # make_grad_safe(self.networks_dict[net_name]['net'].parameters(), -(2 ** 31), 2 ** 31)
                        clip_grad_norm_(self.networks_dict[net_name]['net'].parameters(), 1., torch._six.inf)
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

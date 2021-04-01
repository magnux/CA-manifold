import yaml
from os import path
from torch import optim
from torch.nn import init
from src.optimizers.adamp import AdamP, SGDP


# DEFAULT_CONFIG = path.join(path.dirname(__file__), 'configs/default.yaml')


def load_config(path):
    # with open(DEFAULT_CONFIG, 'r') as f:
    #     config = yaml.load(f)
    with open(path, 'r') as f:
        # config.update(yaml.load(f))
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def build_network(config, class_name, subclass_name, init_scale=None):
    # Get class
    module = __import__('src.networks.%s' % class_name.lower(), fromlist=[subclass_name])
    my_class = getattr(module, subclass_name)

    # Merge args
    kwargs = config['network']['kwargs'].copy()
    kwargs.update(config['data'])
    kwargs.update(config['z_dist'] if 'z_dist' in config else {})

    # Build network
    network = my_class(**kwargs)

    if init_scale is not None:
        init_all(network, init_funcs(init_scale))
    return network


def build_optimizer(network, config):
    optimizer = config['training']['optimizer']
    lr = config['training']['lr']
    equalize_lr = config['training']['equalize_lr']

    if equalize_lr:
        gradient_scales = getattr(network, 'gradient_scales', dict())
        params = get_parameter_groups(network.parameters(), gradient_scales, base_lr=lr)
    else:
        params = network.parameters()

    # Optimizers
    if optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr, betas=(0.5, 0.9), weight_decay=lr * 1e-2, amsgrad=True)
    elif optimizer == 'adamp':
        optimizer = AdamP(params, lr=lr, betas=(0.5, 0.9), weight_decay=lr * 1e-2, nesterov=True)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=lr * 1e-2, nesterov=True)
    elif optimizer == 'sgdp':
        optimizer = SGDP(params, lr=lr, momentum=0.9, weight_decay=lr * 1e-2, nesterov=True)

    return optimizer


def build_lr_scheduler(optimizer, config, last_epoch=0):
    if config['training']['lr_anneal_every'] == 1 and config['training']['lr_anneal'] == 1.0:
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, config['training']['lr'],
                                                     epochs=config['training']['n_epochs'],
                                                     steps_per_epoch=config['training']['steps_per_epoch'],
                                                     last_epoch=last_epoch-1)
    else:
        lr_scheduler = optim.lr_scheduler.OneCycle(
            optimizer, 1e-2,
            step_size=config['training']['lr_anneal_every'],
            gamma=config['training']['lr_anneal'],
            last_epoch=last_epoch-1
        )
    return lr_scheduler


def get_parameter_groups(parameters, gradient_scales, base_lr):
    param_groups = []
    for p in parameters:
        c = gradient_scales.get(p, 1.)
        param_groups.append({
            'params': [p],
            'lr': c * base_lr
        })
    return param_groups


def init_all(model, init_funcs):
    for p in model.parameters():
        init_func = init_funcs.get(len(p.shape), init_funcs["default"])
        init_func(p)


def init_funcs(scale):
    funcs = {
        1: lambda x: init.normal_(x, mean=0., std=scale), # can be bias
        2: lambda x: init.xavier_normal_(x, gain=scale), # can be weight
        3: lambda x: init.xavier_uniform_(x, gain=scale), # can be conv1D filter
        4: lambda x: init.xavier_uniform_(x, gain=scale), # can be conv2D filter
        "default": lambda x: init.constant_(x, scale), # everything else
    }
    return funcs


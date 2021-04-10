import warnings
from torch.optim.lr_scheduler import _LRScheduler


class StepLRm(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr. + MOMENTUM

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLRm(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        self._last_momentum = None
        super(StepLRm, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        lrs = []
        self._last_momentum = []
        for group in self.optimizer.param_groups:
            if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
                lrs.append(group['lr'])
            else:
                lrs.append(group['lr'] * self.gamma)

                if 'betas' in self.optimizer.defaults:
                    beta1, beta2 = group['betas']
                    group['betas'] = (1 - ((1 - beta1) * self.gamma), beta2)
                else:
                    group['momentum'] = 1 - ((1 - beta1) * self.gamma)

            if 'betas' in self.optimizer.defaults:
                self._last_momentum.append(group['betas'])
            else:
                self._last_momentum.append((group['momentum'],))

        return lrs

    def get_last_momentum(self):
        """ Return last computed momentum by current scheduler.
        """
        return self._last_momentum

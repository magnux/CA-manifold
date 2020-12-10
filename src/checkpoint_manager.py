import os
import torch
from os import path
from glob import glob


class CheckpointManager(object):
    def __init__(self, checkpoint_dir='./chkpts'):
        self.module_dict = {}
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.it = 0

    def register_modules(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, it, filename):
        filename = os.path.join(self.checkpoint_dir, filename)
        self.it = it
        outdict = {'it': self.it}
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename):
        filename = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(filename):
            print('=> Loading checkpoint...')
            out_dict = torch.load(filename)
            self.it = out_dict['it']
            for k, v in self.module_dict.items():
                if k in out_dict:
                    out_dict_mod = {}
                    for old_k in out_dict[k]:
                        out_dict_mod[old_k.replace('meta_k', 'dyna_k')] = out_dict[k][old_k]
                    out_dict[k] = out_dict_mod
                    first_key = list(out_dict[k].keys())[0]
                    if first_key in v.state_dict().keys():
                        v.load_state_dict(out_dict[k])
                    elif first_key[:6] == 'module' and first_key[7:] in v.state_dict().keys():
                        print('Warning: loading multigpu module on single gpu or cpu: ', k)
                        out_dict_mod = {}
                        for old_k in out_dict[k]:
                            out_dict_mod[old_k[7:]] = out_dict[k][old_k]
                        v.load_state_dict(out_dict_mod)
                    elif 'module.%s' % first_key in v.state_dict().keys():
                        print('Warning: loading single gpu or cpu module on multigpu: ', k)
                        out_dict_mod = {}
                        for old_k in out_dict[k]:
                            out_dict_mod['module.%s' % old_k] = out_dict[k][old_k]
                        v.load_state_dict(out_dict_mod)
                else:
                    print('Warning: Could not find %s in checkpoint!' % k)

        return self.it

    def load_last(self, prefix):
        last_saves = sorted(glob(self.checkpoint_dir + '/%s_*.pt' % prefix), reverse=True)
        if len(last_saves) > 0:
            return self.load(path.basename(last_saves[0]))
        else:
            return self.load(prefix + '.pt')

    def prune_saves(self, prefix, max_saves=10):
        last_saves = sorted(glob(self.checkpoint_dir + '/%s*.pt' % prefix), reverse=True)
        if len(last_saves) > max_saves:
            for i in range(max_saves, len(last_saves)):
                os.remove(last_saves[i])

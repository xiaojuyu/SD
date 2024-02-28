import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import os

def setRandom(seed, Faster=True):
    if seed is None:
        seed = random.randint(1, 10000)
    print('seed:', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if Faster:
        cudnn.benchmark = True
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

def printArgs(args):
    print('\n'+'='*15+'settings'+'='*15)
    print('pid:', os.getpid())  
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('='*15+'settings'+'='*15+'\n')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
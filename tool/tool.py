import torch
import random
import os
import numpy as np

def load_diff_checkpoint(diff_checkpoint, diff_model):
    state_dict = torch.load(diff_checkpoint)
    weights_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k # 权重dict中的key多了module.，可能是内嵌模型导致的
        weights_dict[new_k] = v
    diff_model.load_state_dict(weights_dict)


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_args(args):
    print('arguments:')
    args_dict = vars(args)
    for arg_name, arg_value in args_dict.items():
        print(f'    {arg_name}: {arg_value}')

def print_transform(transform, note):
    print(note)
    transforms_list = transform.transforms
    for transform in transforms_list:
        print(f'    {transform}')
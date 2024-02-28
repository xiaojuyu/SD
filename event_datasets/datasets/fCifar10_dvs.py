from random import sample
from typing import Callable, Dict, List, Optional, Tuple
from .. import event as ev
from .fEventBase import cifar10_dvsBase
import torch
import numpy as np
import time

class CIFAR10DVS(cifar10_dvsBase):
    """

    Initialize the dataset and overload the function.

    @root: root path of the datasets(str).
    @represent: the representation of the input.
    @step: The number of times an image in SNN is repeatedly entered into the network.
    @folders_names: the download, extract, save folder name.
    @download: whether to download datasets.
    @split: The ratio of datasets.
    @subSet: The subdirectory under convert is used to select Test or Train.
    @extention: the saved file format.
    @ratio: The amount of data used for training or testing, the size is between 0~1, indicating the percentage of data under the subSet.
    @transform: transform x
    @target_transform: transform y

    """
    def __init__(self, 
        root: str, 
        represent:str='timesteps',
        step:int = 100,
        folders_names: list=["oriDownload", "extract", "convert"], 
        download: bool = False, 
        split: list = [9, 1], 
        subSet: str = "Train", 
        extention: str = ".npy", 
        ratio:float = 1.0,
        transform=None, 
        target_transform=None) -> None:

        self.represent = represent.lower()
        self.step = step
        self.ratio =ratio
        print('cifar10dvs', split)

        super().__init__(root, folders_names, download, split, subSet, extention, transform, target_transform)
        
    def make_dataset(self, fsave: str, subSet: str, class_to_idx: Optional[Dict[str, int]] = None, extensions: Optional[Tuple[str, ...]] = None, is_valid_file: Optional[Callable[[str], bool]] = None) -> List[Tuple[str, int]]:
        samples = super().make_dataset(fsave, subSet, class_to_idx, extensions, is_valid_file)
        if self.ratio<1.0:
            np.random.shuffle(samples)
            samples = samples[0:int(self.ratio*len(samples))]
        return samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample:ev.event = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        if self.represent == 'timesteps':
            img = sample.toTimeStep((2, 128, 128, self.step), 'count')
            img = torch.from_numpy(img)
            target = torch.tensor(target)
        elif self.represent in ['est', 'eventcount', 'eventframe', 'voxgrid', 'eventfeature']:
            img = sample.toArray('xytp')
        return img, target
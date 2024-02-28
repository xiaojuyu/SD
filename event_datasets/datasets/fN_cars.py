from .. import event as ev
from .fEventBase import n_carsBase
import torch
import numpy as np
import time

class NCARS(n_carsBase):
    """

    Initialize the dataset and overload the function.

    @root: root path of the datasets(str).
    @represent: the representation of the input.
    @step: The number of times an image in SNN is repeatedly entered into the network.
    @folders_names: the download, extract, save folder name.
    @download: whether to download datasets.
    @subSet: The subdirectory under convert is used to select Test or Train.
    @extention: the saved file format.
    @transform: transform x
    @target_transform: transform y

    """
    def __init__(self, 
        root: str, 
        represent:str='timesteps',
        step:int = 100,
        folders_names: list=["oriDownload", "extract", "convert"], 
        download: bool = False, 
        subSet: str = "Train", 
        extention: str = ".npy", 
        transform=None, 
        target_transform=None) -> None:

        self.represent = represent.lower()
        self.step = step
        super().__init__(root, folders_names, download, subSet, extention, transform, target_transform)
        

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample:ev.event = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        if self.represent == 'timesteps':
            img = sample.toTimeStep((2, 100, 120, self.step), 'count')
            img = torch.from_numpy(img)
            target = torch.tensor(target)
        elif self.represent in ['est', 'eventcount', 'eventframe', 'voxgrid', 'eventfeature']:
            img = sample.toArray('xytp')
        return img, target
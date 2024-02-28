from .. import event as ev
from .fEventBase import n_caltech101Base
import torch
import numpy as np
import time
import os
from PIL import Image

class NCALTECH101(n_caltech101Base):
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
        transform=None, 
        target_transform=None) -> None:

        self.represent = represent.lower()
        self.step = step
        super().__init__(root, folders_names, download, split, subSet, extention, transform, target_transform)
        

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample:ev.event = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        if self.represent == 'timesteps':
            img = sample.toTimeStep((2, 180, 240, self.step), 'count')
            img = torch.from_numpy(img)
            target = torch.tensor(target)
        elif self.represent in ['est', 'eventcount', 'eventframe', 'voxgrid', 'eventfeature']:
            img = sample.toArray('xytp')
        return img, target
    
class CALTECH101_ET(NCALTECH101):
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
        transform=None, 
        target_transform=None) -> None:

        self.eRoot = os.path.join(root, 'NCALTECH101')
        self.tRoot = os.path.join(root, 'CALTECH101')
        super().__init__(self.eRoot, represent, step, folders_names, download, split, subSet, extention, transform, target_transform)
        self.splitCaltech101()

    def splitCaltech101(self):
        tExtractPath = os.path.join(self.tRoot, 'extract')
        tConvertPath = os.path.join(self.tRoot, 'convert')
        eConvertPath = os.path.join(self.eRoot, 'convert')
        if not os.path.exists(tExtractPath) or not os.path.exists(eConvertPath):
            raise FileNotFoundError(f"Couldn't find extract folder.")
        if not os.path.exists(tConvertPath):
                os.makedirs(tConvertPath)
        else:
             return
        eFolderList = os.listdir(eConvertPath)
        for eFolder in eFolderList:
            eFolderPath = os.path.join(eConvertPath, eFolder)
            tFolderPath = os.path.join(tConvertPath, eFolder)
            if not os.path.exists(tFolderPath):
                    os.makedirs(tFolderPath)
            eClassList = os.listdir(eFolderPath)
            for eClass in eClassList:
                eClassPath = os.path.join(eFolderPath, eClass)
                tClassPath = os.path.join(tFolderPath, eClass)
                if not os.path.exists(tClassPath):
                        os.makedirs(tClassPath)
                eImageList = os.listdir(eClassPath)
                for eImage in eImageList:
                    image = os.path.splitext(eImage)[0] + ".jpg"
                    tImageSavePath = os.path.join(tClassPath, image)
                    tImageReadPath = os.path.join(tExtractPath, eClass)
                    tImageReadPath = os.path.join(tImageReadPath, image)
                    if not os.path.exists(tImageReadPath) :
                        raise FileNotFoundError(f"Couldn't find image.")
                    image = Image.open(tImageReadPath)
                    image = image.resize((240,180)).convert("RGB") # 灰度图也转RGB
                    image.save(tImageSavePath)

    # 保存处理后数据
    # def __getitem__(self, index):
    #     ePath, target = self.samples[index]
    #     nPath = ePath.replace("NCALTECH101", "CALTECH101ET")
    #     nPath = nPath.replace(self.extention, ".npy")
    #     directory, filename = os.path.split(nPath)
    #     directory = os.path.join(directory, str(self.step))
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     nPath = os.path.join(directory, filename)
    #     if os.path.exists(nPath):
    #         img = np.load(nPath) 
    #         img = torch.tensor(img)
    #     else:
    #         sample:ev.event = self.loader(ePath)
    #         if self.transform:
    #             sample = self.transform(sample)
    #         if self.target_transform:
    #             target = self.target_transform(target)
    #         eImage = sample.toTimeStep((2, 180, 240, self.step), 'count')
    #         eImage = torch.from_numpy(eImage)
    #         target = torch.tensor(target)
    #         tPath = ePath.replace("NCALTECH101", "CALTECH101")
    #         tPath = tPath.replace(self.extention, ".jpg")
    #         tImage = Image.open(tPath)
    #         tImage = np.array(tImage)
    #         if tImage.ndim == 2:
    #             tImage = tImage.repeat(1,1,3)
    #         tImage = torch.tensor(tImage).permute(2, 0, 1).float()/255.0
    #         t = torch.zeros([1,180,240])
    #         tImage = torch.cat([tImage, t], dim=0)
    #         tImage = tImage.unsqueeze(0)
    #         tImage = tImage.reshape((2,180,240,2))
    #         img = torch.cat([eImage, tImage], dim=3)
    #         np.save(nPath, img.numpy())
    #     return img, target
    
    # 实时处理
    def __getitem__(self, index):
        ePath, target = self.samples[index]
        tPath = ePath.replace("NCALTECH101", "CALTECH101").replace(self.extention, ".jpg")
        sample:ev.event = self.loader(ePath)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        eImage = sample.toTimeStep((2, 180, 240, self.step), 'count') # [2, 180, 240, 100]
        eImage = torch.tensor(eImage)
        target = torch.tensor(target)
        if not os.path.exists(tPath) :
            raise FileNotFoundError(f"Couldn't find image.")
        tImage = Image.open(tPath)
        tImage = np.array(tImage)
        tImage = torch.tensor(tImage, dtype=torch.float32) / 255.0 # [180, 240, 3]
        tImage = tImage.unsqueeze(0).repeat((2,1,1,1)) # [2, 180, 240, 3]
        img = torch.cat([eImage, tImage], dim=3) # [2, 180, 240, 103]
        return img, target
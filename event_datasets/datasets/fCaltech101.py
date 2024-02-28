from random import sample
from typing import Callable, Dict, List, Optional, Tuple
from abc import abstractmethod
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, extract_archive ,calculate_md5
from torch.utils.data import Dataset
import os
import math 
import torch
import numpy as np
import time
from PIL import Image

class Caltech101(Dataset):

    def __init__(self, 
        root: str, 
        folders_names: list=["oriDownload", "extract", "convert"], 
        split: list = [9, 1], 
        subset: str = "Train", 
        ratio:float = 1.0,
        extention: str = ".jpg", 
        transform=None, 
        target_transform=None,
        download=False,
        reload=False) -> None:

        self.root = root
        self.folders_names = folders_names
        self.split = split
        self.subset = subset
        self.ratio =ratio
        self.extention =extention
        self.transform = transform
        self.target_transform = target_transform

        
        self.fdownload = os.path.join(self.root, 'caltech101')
        self.fextracted = os.path.join(self.fdownload, '101_ObjectCategories')
        self.fsave = os.path.join(self.root, 'save')
        self.ftrain = os.path.join(self.fsave, 'Train')
        self.ftest = os.path.join(self.fsave, 'Test')
        self.fvalidation = os.path.join(self.fsave, 'Validation')

        if download:
            reload = self.download(self.fdownload)
        
        self.fread = os.path.join(self.fsave, subset)
        if reload or not os.path.exists(self.fread):
            self.splitToTrainAndTest(self.fextracted,self.fsave)
        
        self.samples = self.make_dataset_ratio(self.fread)
    
    def download(self,path):
        return False

    def splitSet(self, all_img, split, fsave, classname):
        """
        split specific class's imgs.
        """
        assert len(split) > 1, f"split error:{split}, we need at least 2 number"
        
        t = float(sum(split))
        split = [math.ceil(len(all_img)/t*i) for i in split]
        split[1] += split[0]
        train_path = os.path.join(fsave, "Train", classname)
        test_path = os.path.join(fsave, "Test", classname)
        if len(split) == 2:
            print(f"split {classname} into Train and Test set...")
            train_img, test_img = np.split(np.random.permutation(all_img), 
                    [split[0]])
            return {train_path: train_img, test_path: test_img}
        elif len(split) == 3:
            print("split data into Train, Validation and Test set...")
            train_img, validation_img, test_img = np.split(np.random.permutation(all_img), 
                    [split[0], split[1]])
            validation_path = os.path.join(fsave, "Validation", classname)
            return {train_path: train_img, validation_path:validation_img, test_path: test_img}

    def find_classes_FromFolder(self, directory: str, filter_dir:Optional[Callable]=lambda x: True) -> Dict[str, int]:
        classes = [entry.name for entry in os.scandir(directory) if entry.is_dir() and filter_dir(entry)]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return class_to_idx

    def splitToTrainAndTest(self,fextracted,fsave):
        class_to_idx = self.find_classes_FromFolder(fextracted)
        for name in class_to_idx.keys():
            if(name == 'Faces'):
                continue
            ex_class_path = os.path.join(fextracted, name)
            if os.path.isdir(ex_class_path):
                all_img = os.listdir(ex_class_path)
                for img in all_img:
                    if(img.split('.')[-1]!='jpg'):
                        all_img.remove(img)
                path_to_imgs = self.splitSet(all_img, self.split, fsave, name)
            for targetClassPath, setImgs in path_to_imgs.items():
                if not os.path.exists(targetClassPath):
                    os.makedirs(targetClassPath)
                for idx, img in enumerate(setImgs):
                    image = Image.open(os.path.join(ex_class_path, img))
                    # image.save(os.path.join(targetClassPath, img),self.extention)
                    image.save(os.path.join(targetClassPath, img))

    def make_dataset(self, fread, class_to_idx: Optional[Dict[str, int]] = None):
        directory = os.path.expanduser(fread)

        if class_to_idx is None:
            class_to_idx = self.find_classes_FromFolder(directory)
        instances = []
        available_classes = set()
        for target_class, class_index in class_to_idx.items():
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in os.walk(target_dir, followlinks=True):
                for fname in fnames:
                    instances.append((os.path.join(root, fname), class_index))
                    if target_class not in available_classes:
                        available_classes.add(target_class)
        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            raise FileNotFoundError(msg)
        return instances

    def make_dataset_ratio(self, fread: str, class_to_idx: Optional[Dict[str, int]] = None) -> List[Tuple[str, int]]:
        samples = self.make_dataset(fread, class_to_idx)
        if self.ratio<1.0:
            np.random.shuffle(samples)
            samples = samples[0:int(self.ratio*len(samples))]
        return samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)
# rely on torch, torchvision

from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, extract_archive ,calculate_md5
from torch.utils.data import Dataset
from .. import event as ev
import os
import numpy as np
import math 

'''
when free, we can provide return classes for functions such as resource. And check function for restructureAndConvert.
'''

def multi_download(func):
    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor
    from functools import wraps
    @wraps(func)
    def decorated(*args, **kwargs):
        pass
        return func(*args, **kwargs)
    return decorated


class eventBase(Dataset):
    def __init__(self, 
        root:str, 
        fname:list, 
        download: bool,
        
        loader:Callable,
        subSet:str="Train", 
        extention:str=".npy",
        transform=None,
        target_transform=None,
        ) -> None:
        """
        wrapper download, and basic function like read datasets, getitem, len.
        @root: root path of the datasets(str).
        @fname: the download, extract, save folder name.
        @loader: loader data for format of extention.
        @download: whether to download datasets.
        @extention: the saved file format.
        @subSet
        @transform: transform x
        @target_transform: transform y
        @multi_process: multiprogress and Breakpoint continuingly(not implement yet).
        you need to implement downloadResource, restructureAndConvert.
        this will help you download from net and 
        restructure the datasets into specific format(extention), and the structured as follow:
        directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext
        the basic format of sample as (t, x, y, p), label as (i).

        if you want other, or save other format in disk, please implement restructureAndConvert first,
        and reload function: make_dataset, you can save datasets in one file, or other sample format.
        if there are subsequent processing of sample data, like EST, FrameCount, etc., 
        you can reload function: __getitem__.
        ori_format is (t, x, y, p), label: (i).
        the dataflow is: download -> restructureAndConvert(ori_format) -> make_datasets -> __getitem__ -> out
        '''
    """

        self.root = root
        assert len(fname) == 2 or len(fname) == 3, f"there are {len(fname)} folder names. the download and save name must be different, and no extra."
        assert subSet in ["Train", "Test", "Validation"], 'the subSet must in ["Train", "Test", "Validation"]'
        
        self.fdownload = os.path.join(self.root, fname[0])
        self.fextract = os.path.join(self.root, fname[-2])
        self.fsave = os.path.join(self.root, fname[-1])
        self.extention = extention
        
        reload = False
        if download:
            reload = self.download(self.fdownload, self.fextract)
        if reload or not os.path.exists(self.fsave):
            self.restructureAndConvert(self.fextract, self.fsave, self.extention)
        
        self.samples = self.make_dataset(self.fsave,subSet,extensions=self.extention)
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.fsave, subSet) + "\n"
                                "Supported extensions are: " + self.extention))

    
    @abstractmethod 
    def downloadResource(self)->list:
        """
        @return: a list of download resorce, which will be called in download function.
        or raise error when you don't know how to implement it and set download False.
        @rtype:[(str,str,str)]: [(filename, url, md5)]
        """
        raise NotImplementedError

    @abstractmethod 
    def restructureAndConvert(self, fextracted:str, fsave:str, ext:str)->None:
        """
        read from the original directory structure in source(path extracted), 
        and split into train/validate/test set(if not),
        and convert to other format and save.

        restructure the datasets into specific format(extention), and the structured as follow:
        fsave/
            ├── Train
            |   ├── class_x
            |   │   ├── xxx.ext
            |   │   ├── xxy.ext
            |   │   ├── ...
            |   │   └── xxz.ext
            |   └── class_y
            |       ├── 123.ext
            |       ├── nsdf3.ext
            |       ├── ...
            |       └── asd932_.ext
            ├── Test
            |   ├── ...
        the basic format of sample as (t, x, y, p), label as (i).
        make sure the fsave will be structured as required. and save single event file into extention file.
        read as list [t, x, y, p]
        """
        raise NotImplementedError

      
    def make_dataset(
        self,
        fsave: str,
        subSet: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.
        OR when you save all the datasets in one file, thus you can read one-time.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(os.path.join(fsave, subSet))

        if class_to_idx is None:
            class_to_idx = self.find_classes_FromFolder(directory)

        if extensions is None == is_valid_file is None:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)

        instances = []
        available_classes = set()
        for target_class, class_index in class_to_idx.items():
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in os.walk(target_dir, followlinks=True):
                for fname in fnames:
                    if is_valid_file(fname):
                        instances.append((os.path.join(root, fname), class_index))

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def find_classes_FromFolder(self, directory: str, filter_dir:Optional[Callable]=lambda x: True) -> Dict[str, int]:
        """
        Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to adapt to a different dataset directory structure.
        When you need a subset or exclude some files, filter_dir may help you.

        Args:
            directory(str): Path to find. Root directory path.
            filter_dir: func(dir:os.DirEntry)->bool  a function to filter the directories of classes. return True if need

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            Dict class_name : idx
        
        """
        classes = [entry.name for entry in os.scandir(directory) if entry.is_dir() and filter_dir(entry)]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _download(self, root:str, fextracted:str, filename:str, url:str, md5:str)->None:
        """
        A wrapper of download_and_extract_archive
        for errors:
        it will print message if files can not be extracted, which is ValueError
        otherwise raise RuntimeError.
        """
        try:
            print(f'Download [{filename}] from [{url}] to [{root}]')
            download_and_extract_archive(url, root, fextracted, filename=filename, md5=md5)
        except ValueError as ve:
            print(ve)
        except BaseException:
            raise RuntimeError('download error, please download manually!')

    def download(self, fdownload:str, fextracted:str = None) -> bool:
        """
        If the dataset doesn't exist, download it, otherwise check for corrections.
        @return: return True if download files from internet, otherwise False.
        """
        reload = False
        if not os.path.exists(fdownload):
            os.makedirs(fdownload)
        for filename, url, md5 in self.downloadResource():
            filepath = os.path.join(fdownload, filename)
            print(f"check file:{filepath}")
            if not check_integrity(filepath, md5):
                reload = True
                if os.path.exists(filepath):
                    print(f'The file:[{filepath}] is incorrect.')
                    os.remove(filepath)
                else:
                    print(f'The file:[{filepath}] does not exist.')
                print(f'Start to download and extract:[{filepath}].')
                self._download(fdownload, fextracted, filename, url, md5)
            if not os.path.exists(os.path.join(fextracted, os.path.splitext(filename)[0])):
                print(f'Start to extract:{filepath} to {fextracted}.')
                extract_archive(filepath, fextracted)
        return reload
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * 4 + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> str:
        return ""


class cifar10_dvsBase(eventBase):
    """
    basic cifar10_dvs, will download, restructure, and read sample, label as (t, x, y, p), i.
    since cifar10_dvs is not splited into subset, split parameter will transform the ratio.

    we implement download and restructureAndConvert. so all the data get the same directory structure and format.
    for more process, reload make_datasets, __getitem__

    @root: root path of the datasets(str).
    @folders_names: the download, extract, save folder name.
    @download: whether to download datasets.

    @split: The ratio of datasets
    @subSet
    @extention: the saved file format.
    @transform: transform x
    @target_transform: transform y

    
    """
    def __init__(self,
        root:str, 
        folders_names:list=["oriDownload", "extract", "convert"],
        download: bool=False,
        split: list=[9, 1],
        subSet: str = "Train",
        extention:str=".npy",
        transform=None, 
        target_transform=None
        # saver:Callable=np.save
        ) -> None:

        self.height =128
        self.width = 128
        self.split = split
        def myloader(path)->ev.event:
            return ev.load(path, extention)
        def mysaver(path, data:ev.event):
            ev.save(path, data, extention)
        self.saver = mysaver
        print('cifar10dvsbase', split)
        super().__init__(root, folders_names, download, myloader, subSet, extention, transform, target_transform)
    
    
    def loadaerdat(self, filename)->ev.event:
        '''
        read the events
        '''
        return ev.loadaerdat(filename, camera='DVS128')

    def splitSet(self, all_img, split, fsave, classname):
        """
        split specific class's imgs.
        """
        print('splitSet', split)
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

    def restructureAndConvert(self, fextracted:str, fsave:str, ext:str)->None:
        """
        restructure the directory of datasets, since cifar10_dvs is not splited, split by yourself when restructure.
        and convert into format of ext.
        fills will convert from fextarcted to fsave.
        """
        class_to_idx = self.find_classes_FromFolder(fextracted)

        for name in class_to_idx.keys():
            ex_class_path = os.path.join(fextracted, name)
            if os.path.isdir(ex_class_path):
                all_img = os.listdir(ex_class_path)
                path_to_imgs = self.splitSet(all_img, self.split, fsave, name)
            print("start to convert ", name, "...\n")
            for targetClassPath, setImgs in path_to_imgs.items():
                for idx, img in enumerate(setImgs):
                    print(f"\033[1AImage's idx:{idx}\033[K")
                    if not os.path.exists(targetClassPath):
                        os.makedirs(targetClassPath)
                    targetImgPath = os.path.join(targetClassPath, os.path.splitext(img)[0]+ext)
                    if not os.path.exists(targetImgPath):
                        evdata = self.loadaerdat(os.path.join(ex_class_path, img))
                        self.saver(targetImgPath, evdata.removeInvalid(dim_x=self.width, dim_y=self.height))
    
    def downloadResource(self) -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        if you can not found the resource. just raise error.
        '''
        return [
            ('airplane.zip', 'https://ndownloader.figshare.com/files/7712788', '0afd5c4bf9ae06af762a77b180354fdd'),
            ('automobile.zip', 'https://ndownloader.figshare.com/files/7712791', '8438dfeba3bc970c94962d995b1b9bdd'),
            ('bird.zip', 'https://ndownloader.figshare.com/files/7712794', 'a9c207c91c55b9dc2002dc21c684d785'),
            ('cat.zip', 'https://ndownloader.figshare.com/files/7712812', '52c63c677c2b15fa5146a8daf4d56687'),
            ('deer.zip', 'https://ndownloader.figshare.com/files/7712815', 'b6bf21f6c04d21ba4e23fc3e36c8a4a3'),
            ('dog.zip', 'https://ndownloader.figshare.com/files/7712818', 'f379ebdf6703d16e0a690782e62639c3'),
            ('frog.zip', 'https://ndownloader.figshare.com/files/7712842', 'cad6ed91214b1c7388a5f6ee56d08803'),
            ('horse.zip', 'https://ndownloader.figshare.com/files/7712851', 'e7cbbf77bec584ffbf913f00e682782a'),
            ('ship.zip', 'https://ndownloader.figshare.com/files/7712836', '41c7bd7d6b251be82557c6cce9a7d5c9'),
            ('truck.zip', 'https://ndownloader.figshare.com/files/7712839', '89f3922fd147d9aeff89e76a2b0b70a7')
        ]
 

class mnist_dvsBase(eventBase):
    """
    basic mnistdvs, will download, restructure, and read sample, label as (t, x, y, p), i.
    since mnistdvs is not splited into subset, split parameter will transform the ratio.

    we implement download and restructureAndConvert. so all the data get the same directory structure and format.
    for more process, reload make_datasets, __getitem__

    @root: root path of the datasets(str).
    @scale: The scale of the image in the MNISTDVS dataset.
    @folders_names: the download, extract, save folder name.
    @download: whether to download datasets.
    @split: The ratio of datasets
    @subSet: The subdirectory under convert is used to select Test or Train.
    @extention: the saved file format.
    @transform: transform x
    @target_transform: transform y

    
    """
    def __init__(self,
        root:str, 
        scale: str = "scale16",
        folders_names:list=["oriDownload", "extract", "convert"],
        download: bool=False,
        split: list=[9, 1],
        subSet: str = "Train",
        extention:str=".npy",
        transform=None, 
        target_transform=None
        # saver:Callable=np.save
        ) -> None:

        self.height =128
        self.width = 128
        self.split = split
        myloader = lambda x: ev.load(x, extention)
        def mysaver(path, data:ev.event):
            ev.save(path, data, extention)
        self.saver = mysaver
        self.scale = scale
        super().__init__(root, folders_names, download, myloader, subSet, extention, transform, target_transform)
    
    def make_dataset(self, fsave: str, subSet: str, class_to_idx: Optional[Dict[str, int]] = None, extensions: Optional[Tuple[str, ...]] = None, is_valid_file: Optional[Callable[[str], bool]] = None) -> List[Tuple[str, int]]:
        fsave = os.path.join(fsave, self.scale)
        return super().make_dataset(fsave, subSet, class_to_idx, extensions, is_valid_file)
   
    def loadaerdat(self, filename)->ev.event:
        '''
        read the event data and return (ts, xpos, ypos, pol) 4-tuple of lists containing data of all events
        '''
        return ev.loadaerdat(filename, camera = 'DVS128')

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

    def restructureAndConvert(self, fextracted:str, fsave:str, ext:str)->None:
        """
        restructure the directory of datasets, since mnistdvs is not splited, split by yourself when restructure.
        and convert into format of ext.
        fills will convert from fextarcted to fsave.
        """
        # scale = self.scale
        # scales = self.find_classes_FromFolder(fextracted)
        scales = ['scale4','scale8','scale16']
        for scale in scales:
            fsave_scale = os.path.join(fsave, scale)
            class_to_idx = self.find_classes_FromFolder(fextracted)

            for name in class_to_idx.keys():
                ex_class_path = os.path.join(fextracted, name)
                ex_class_scale_path = os.path.join(ex_class_path, scale)
                if os.path.isdir(ex_class_scale_path):
                    all_img = os.listdir(ex_class_scale_path)
                    path_to_imgs = self.splitSet(all_img, self.split, fsave_scale, name)
                print("start to convert ", name, "...\n")
                for targetClassPath, setImgs in path_to_imgs.items():
                    for idx, img in enumerate(setImgs):
                        print(f"\033[1AImage's idx:{idx}\033[K")
                        if not os.path.exists(targetClassPath):
                            os.makedirs(targetClassPath)
                        targetImgPath = os.path.join(targetClassPath, os.path.splitext(img)[0]+ext)
                        if not os.path.exists(targetImgPath):
                            evdata = self.loadaerdat(os.path.join(ex_class_scale_path, img))
                            self.saver(targetImgPath, evdata.removeInvalid(dim_x=self.width, dim_y=self.height))


    def downloadResource(self) -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        if you can not found the resource. just raise error.
        '''
        return [
            ('grabbed_data0.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data0.zip', '451C1D2F6B195CE96D1DD478EE8AAF6E'.lower()),
            ('grabbed_data1.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data1.zip', '22259AE8871800B5EECF9E0EB8146EA5'.lower()),
            ('grabbed_data2.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data2.zip', '6D3AD96939B702F095D027CE89F1297C'.lower()),
            ('grabbed_data3.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data3.zip', 'E4833915A1514598C4AE046E5E09CF93'.lower()),
            ('grabbed_data4.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data4.zip', 'E183266A0B670B5336BA4D39D4BD30AE'.lower()),
            ('grabbed_data5.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data5.zip', 'ABF900993C268A763639C3074B6F92BB'.lower()),
            ('grabbed_data6.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data6.zip', '971DB9A99E95B408837518C7727A223C'.lower()),
            ('grabbed_data7.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data7.zip', 'A3CB115F19A736F4FC3590F36012BC87'.lower()),
            ('grabbed_data8.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data8.zip', 'FE5DA6C6F559ABF397A32439A6C1CD3F'.lower()),
            ('grabbed_data9.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data9.zip', '73A3007C1C1AB4BFAD9CE07EA76D540D'.lower())
        ]


class n_carsBase(eventBase):
    """
    basic ncars, will download, restructure, and read sample, label as (t, x, y, p), i.
    The ncars has been divided into training set and testing set.

    we implement download and restructureAndConvert. so all the data get the same directory structure and format.
    for more process, reload make_datasets, __getitem__

    @root: root path of the datasets(str).
    @folders_names: the download, extract, save folder name.
    @download: whether to download datasets.
    @subSet: The subdirectory under convert is used to select Test or Train.
    @extention: the saved file format.
    @transform: transform x
    @target_transform: transform y

    
    """
    def __init__(self,
        root:str, 
        folders_names:list=["oriDownload", "extract", "convert"],
        download: bool=False,
        subSet: str = "Train",
        extention:str=".npy",
        transform=None, 
        target_transform=None
        # saver:Callable=np.save
        ) -> None:

        self.height = 100
        self.width = 120
        myloader = lambda x: ev.load(x, extention)
        def mysaver(path, data:ev.event):
            ev.save(path, data, extention)
        self.saver = mysaver
        super().__init__(root, folders_names, download, myloader, subSet, extention, transform, target_transform)

    def loadaerdat(self, filename)->ev.event:
        '''
        read the event data and return (ts, xpos, ypos, pol) 4-tuple of lists containing data of all events
        '''
        return ev.loadaerdat(filename, camera='N')

    def restructureAndConvert(self, fextracted:str, fsave:str, ext:str)->None:
        """
        ncars has been divided, we directly convert into format of ext in the corresponding directory.
        fills will convert from fextarcted to fsave.
        """
        fextracted = os.path.join(fextracted, "n-cars")
        test_or_train = ['Test','Train']
        for tt in test_or_train:
            fsave_tt = os.path.join(fsave, tt)
            # fextracted_class = os.path.join(fextracted, tt)
            class_to_idx = self.find_classes_FromFolder(fextracted)

            for name in class_to_idx.keys():
                ex_class_path = os.path.join(fextracted, name)
                fsave_tt_class = os.path.join(fsave_tt, name)
                if os.path.isdir(ex_class_path):
                    all_img = os.listdir(ex_class_path)
                for idx, img in enumerate(all_img):
                    print(f"\033[1AImage's idx:{idx}\033[K")
                    if not os.path.exists(fsave_tt_class):
                        os.makedirs(fsave_tt_class)
                    targetImgPath = os.path.join(fsave_tt_class, os.path.splitext(img)[0]+ext)
                    if not os.path.exists(targetImgPath):
                        evdata = self.loadaerdat(os.path.join(ex_class_path, img))
                        self.saver(targetImgPath, evdata.removeInvalid(dim_x=self.width, dim_y=self.height))
   
    def downloadResource(self) -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        if you can not found the resource. just raise error.
        '''
        return [
            # ('n-cars.zip', 'https://figshare.com/ndownloader/files/34573463', 'D1DEF7EEA897126269D19C570DF3550A'.lower())
            ('n-cars.zip', 'https://figshare.com/ndownloader/files/34573463', '914819b99dc7a6564326ba627919bf2e')
        ]


class n_caltech101Base(eventBase):
    """
    basic ncaltech101, will download, restructure, and read sample, label as (t, x, y, p), i.
    since ncaltech101 is not splited into subset, split parameter will transform the ratio.

    we implement download and restructureAndConvert. so all the data get the same directory structure and format.
    for more process, reload make_datasets, __getitem__

    @root: root path of the datasets(str).
    @folders_names: the download, extract, save folder name.
    @download: whether to download datasets.
    @split: The ratio of datasets
    @subSet: The subdirectory under convert is used to select Test or Train.
    @extention: the saved file format.
    @transform: transform x
    @target_transform: transform y

    
    """
    def __init__(self,
        root:str, 
        folders_names:list=["oriDownload", "extract", "convert"],
        download: bool=False,
        split: list=[9, 1],
        subSet: str = "Train",
        extention:str=".npy",
        transform=None, 
        target_transform=None
        # saver:Callable=np.save
        ) -> None:

        self.height = 180
        self.width = 240
        self.split = split
        myloader = lambda x: ev.load(x, extention)
        def mysaver(path, data:ev.event):
            ev.save(path, data, extention)
        self.saver = mysaver
        super().__init__(root, folders_names, download, myloader, subSet, extention, transform, target_transform)

    def loadaerdat(self, filename)->ev.event:
        '''
        read the event data and return (ts, xpos, ypos, pol) 4-tuple of lists containing data of all events
        '''
        return ev.loadaerdat(filename, camera='N101')

    def splitSet(self, all_img, split, fsave, classname):
        """
        split specific class's imgs.
        """
        assert len(split) > 1, f"split error:{split}, we need at least 2 number"
        
        t = float(sum(split))
        split = [math.ceil(len(all_img)/t*i) for i in split]
        split[1] += split[0]
        print('all imgs', len(all_img))
        print('split:', split)
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

    def restructureAndConvert(self, fextracted:str, fsave:str, ext:str)->None:
        """
        restructure the directory of datasets, since ncaltech101 is not splited, split by yourself when restructure.
        and convert into format of ext.
        fills will convert from fextarcted to fsave.
        """
        fextracted = os.path.join(fextracted, 'Caltech101')
        class_to_idx = self.find_classes_FromFolder(fextracted)

        for name in class_to_idx.keys():
            ex_class_path = os.path.join(fextracted, name)
            if os.path.isdir(ex_class_path):
                all_img = os.listdir(ex_class_path)
                path_to_imgs = self.splitSet(all_img, self.split, fsave, name)
            print("start to convert ", name, "...\n")
            for targetClassPath, setImgs in path_to_imgs.items():
                for idx, img in enumerate(setImgs):
                    print(f"\033[1AImage's idx:{idx}\033[K")
                    if not os.path.exists(targetClassPath):
                        os.makedirs(targetClassPath)
                    targetImgPath = os.path.join(targetClassPath, os.path.splitext(img)[0]+ext)
                    if not os.path.exists(targetImgPath):
                        evdata = self.loadaerdat(os.path.join(ex_class_path, img))
                        self.saver(targetImgPath, evdata.removeInvalid(dim_x=self.width, dim_y=self.height))
    

    def downloadResource(self) -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        if you can not found the resource. just raise error.
        '''
        return [
            ('Caltech101.zip', 'https://md-datasets-public-files-prod.s3.eu-west-1.amazonaws.com/c708f296-c555-4509-8ca9-3ff2504679ee', '66201824eabb0239c7ab992480b50ba3')
        ]
 
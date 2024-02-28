import pickle
import os
import re
from PIL import Image

def trans_to_images(path):
    imagesPath = os.path.join(path, "images")
    if not os.path.exists(imagesPath):
            os.makedirs(imagesPath)
    for i in range(10):
        classPath = os.path.join(imagesPath, str(i))
        if not os.path.exists(classPath):
                os.makedirs(classPath)
    folderPath = os.path.join(path, "oriDownload")
    filesList = os.listdir(folderPath)
    for fileName in filesList:
        filePath = os.path.join(folderPath, fileName)
        if re.match(".*_batch.*",fileName) and os.path.isfile(filePath):
            with open(filePath, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                for index, image in enumerate(dict[b'data']):
                     imagePath = os.path.join(imagesPath, str(dict[b'labels'][index]))
                     imagePath = os.path.join(imagePath, dict[b'filenames'][index].decode())
                     image = image.reshape(3,1024).T
                     image = image.reshape(32,32,3)
                     image = Image.fromarray(image)
                     image.save(imagePath)


if __name__ == '__main__':
    path = "/home/xjy/datasets/CIFAR10"
    trans_to_images(path)
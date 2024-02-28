import sys
sys.path.append("../SNN/") 
from event_datasets import event as ev
import torch
import numpy as np
import os
from PIL import Image

def event_to_array(event, dim):
    num_voxels = int(2 * np.prod(dim))
    vox_ec = event[0].new_full([num_voxels,], fill_value=0)        
    H, W = dim 
    x, y, t, p = event.T
    idx = x + W*y + W*H*p
    val = torch.zeros_like(x) + 1
    vox_ec.put_(idx.long(), val, accumulate=True)
    vox_ec = vox_ec.view(2, H, W)
    vox_ec = vox_ec.sum(dim=0)
    return vox_ec

def trans_to_images(path):
    imagesPath = os.path.join(path, "images")
    if not os.path.exists(imagesPath):
            os.makedirs(imagesPath)
    folderPath = os.path.join(path, "extract")
    classList = os.listdir(folderPath)
    for className in classList:
        imagePath = os.path.join(imagesPath, className)
        if not os.path.exists(imagePath):
                os.makedirs(imagePath)
        classPath = os.path.join(folderPath, className)
        allSpikingImages = os.listdir(classPath)
        for index, spikingImage in enumerate(allSpikingImages):
            spikingImagePath = os.path.join(classPath, spikingImage)
            eventData = ev.loadaerdat(spikingImagePath, camera='DVS128')
            image = torch.tensor(eventData.toArray('xytp'))
            image = event_to_array(image, [128,128]).unsqueeze(0).repeat(3,1,1).permute(1,2,0).numpy()
            image = Image.fromarray(np.uint8(image))
            image.save(os.path.join(imagePath, os.path.splitext(spikingImage)[0] + ".png"))

if __name__ == '__main__':
    path = "/home/xjy/datasets/CIFAR10DVS"
    trans_to_images(path)
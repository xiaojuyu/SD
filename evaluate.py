import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader
import sys
import argparse
import random
import metric.pytorch_ssim

from PIL import Image
from model.snn_diffusion import *
from metric.Fid_score import *
from metric.IS_score import *
from tool.tool import load_diff_checkpoint, setup_seed, print_args
from tool.load_dataset_snn import load_dataset, image_to_step, image_to_event, event_normalize
from config.config import get_network_config, get_diffusion_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=80, type=int)
parser.add_argument('--print_freq', default=40, type=int)
parser.add_argument('--workers', default=12, type=int, metavar='N',help='number of data loading workers (default: 10)')
parser.add_argument('--dataset_name', default='CIFAR10', type=str)
parser.add_argument('--data_path', default='/home/xjy/datasets', type=str)
# parser.add_argument('--result_path', default='./result1/sample_images_ddpm', type=str)
parser.add_argument('--result_path', default='./result3/sample_images_ddim', type=str)
parser.add_argument('--real_images_path', default='/home/xjy/RealImages/cifar10.npy', type=str)
parser.add_argument('--all_images_path', default='./result3/sample_images_ddim/sample_images.npy', type=str)
parser.add_argument('--model_checkpoint', default='/home/xjy/experiment/CIFAR10/video1.0/step4/result2/ckp_latest.pth', type=str,action='store', dest='model_checkpoint',help='The path of checkpoint, if use checkpoint')
# Common
parser.add_argument('--img_size', default=[10,3,32,32], help='image size [B, C, H, W]')
parser.add_argument('--step',type=int,default=4)
# Spiking UNet
parser.add_argument('--c_in', default=3, type=int)
parser.add_argument('--c_out', default=3, type=int)
parser.add_argument('--c_base', default=128, type=int, help='base channel of UNet')
parser.add_argument('--c_mult', default=[1, 2, 4, 8], help='channel multiplier')
parser.add_argument('--attn', default=[1,], help='add attention to these levels')
parser.add_argument('--num_res_blocks', default=2, type=int, help='# resblock in each level')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of resblock')
# Diffusion
parser.add_argument('--noise_steps', default=1000, type=int, help='total diffusion steps')
parser.add_argument('--beta_start', default=1e-4, type=float, help='start beta value')
parser.add_argument('--beta_end', default=0.02, type=float, help='end beta value')
args = parser.parse_args()

def evaluate_diffusion_sample_test(diffusion, model, test_loader, device, result_path):
    model.eval()
    for i, (x, labels) in enumerate(test_loader):
        x = x.float().to(device)
        images_forward_x, images_backward_x = diffusion.sample_test(x, model, device, skip=2, eta_flag='ddim')
        break

    for i in range(len(images_forward_x)):
        forward_x = images_forward_x[i][0]
        forward_x = (forward_x.clamp(-1, 1) + 1) / 2
        backward_x = images_backward_x[i][0]
        backward_x = (backward_x.clamp(-1, 1) + 1) / 2

        forward_x = forward_x / forward_x.max() * 255.0
        forward_x = forward_x.permute(1,2,0).cpu().numpy()
        forward_x = Image.fromarray((forward_x).astype('uint8'))
        forward_x_path = os.path.join(result_path, 'forward_x{}.png'.format(i))
        forward_x.save(forward_x_path)

        backward_x = backward_x / backward_x.max() * 255.0
        backward_x = backward_x.permute(1,2,0).cpu().numpy()
        backward_x = Image.fromarray((backward_x).astype('uint8'))
        backward_x_path = os.path.join(result_path, 'backward_x{}.png'.format(i))
        backward_x.save(backward_x_path)

        print('i:{}'.format(i))

def evaluate_diffusion_sample_ddpm(diffusion, model, result_path, n = 100):
    images = diffusion.sample_ddpm(model, device, n = n)
    for i in range(n):
        image = images[i].permute(1,2,0).repeat(1,1,3).cpu().numpy()
        image = Image.fromarray((image).astype('uint8'))
        image_path = os.path.join(result_path, 'image{}.png'.format(i))
        image.save(image_path)

def evaluate_diffusion_sample_ddim(diffusion, model, result_path, n = 256):
    # images = diffusion.sample_ddim(model, device, skip=1, eta_flag='ddpm', n=n)
    images = diffusion.sample_ddim(model, device, skip=2, eta_flag='ddim', n=n)
    for i in range(n):
        image = images[i].permute(1,2,0).cpu().numpy()
        image = Image.fromarray((image).astype('uint8'))
        image_path = os.path.join(result_path, 'image{}.png'.format(i))
        image.save(image_path)

def evaluate_diffusion_calculate_score(diffusion, model, test_loader, all_images_path, real_images_path = None):
    all_images = []
    real_images = []
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).type(torch.cuda.FloatTensor)
    if real_images_path == None:
        for i, (x, labels) in enumerate(test_loader):
            print(f'load real images:{i}. Current time: {datetime.datetime.now()}.')
            x = (x + 1) / 2
            x = up(x)
            x = x.cpu().numpy()
            x=np.transpose(x,(0,2,3,1))
            if i == 0:
                real_images = x
            else:
                real_images = np.concatenate((real_images,x),axis=0)
        np.save('/home/xjy/RealImages/cifar10.npy', real_images)
    else:
        real_images = np.load(real_images_path)
    for i in range(100):
        print(f'generate images:{i}. Current time: {datetime.datetime.now()}.')
        # x = diffusion.sample(model, device, skip=1, eta_flag='ddpm', n=100)
        x = diffusion.sample(model, device, skip=2, eta_flag='ddim', n=100)
        x = (x + 1) / 2
        x = up(x)
        x = x.cpu().numpy()
        x=np.transpose(x,(0,2,3,1))
        if i == 0:
            all_images = x
        else:
            all_images = np.concatenate((all_images,x),axis=0)
        np.save(all_images_path, all_images)
        # break
    evaluate_diffusion_calculate_score_npy(all_images, real_images)

def evaluate_diffusion_calculate_score_npy(all_images, real_images):
    Fid = calculate_fid(all_images, real_images, use_multiprocessing=False, batch_size=4)
    print(f'Fid:{Fid}')
    all_images = np.transpose(all_images,(0,3,1,2))
    Is,_ = inception_score(all_images, cuda=True, batch_size=32, resize=True, splits=4)
    print(f'Is:{Is}')

def evaluate_diffusion_calculate_score_path(all_images_path, real_images_path):
    all_images = np.load(all_images_path)
    real_images = np.load(real_images_path)
    Fid = calculate_fid(all_images, real_images, use_multiprocessing=False, batch_size=4)
    print(f'Fid:{Fid}')
    all_images = np.transpose(all_images,(0,3,1,2))
    Is,_ = inception_score(all_images, cuda=True, batch_size=32, resize=True, splits=4)
    print(f'Is:{Is}')

def calculate_score(all_images_path, real_images_path):
    all_images = np.load(all_images_path)
    real_images = np.load(real_images_path)
    for i in range(5):
        all_images_temp = all_images[:1000 + 1000 * i]
        print(f'i: {i}')
        evaluate_diffusion_calculate_score_npy(all_images_temp, real_images)

if __name__ == '__main__':
    print_args(args)

    setup_seed(args.seed)

    if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)

    train_loader, test_loader = load_dataset(dataset_name = args.dataset_name, data_path = args.data_path, batch_size = args.batch_size, step = args.step, workers = args.workers)

    model = SpikingUNet(noise_steps = args.noise_steps, c_in=args.c_in, c_out=args.c_out, c_base=args.c_base, c_mult=args.c_mult, attn=args.attn, num_res_blocks=args.num_res_blocks, dropout=args.dropout, timestep=args.step)
    if args.model_checkpoint != None:
        load_diff_checkpoint(args.model_checkpoint, model)
    model = torch.nn.DataParallel(model) 
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    mse = nn.MSELoss()

    diffusion = SpikingDiffusion(noise_steps = args.noise_steps, beta_start = args.beta_start, beta_end = args.beta_end, img_size = args.img_size, device = device)

    # evaluate_diffusion_sample_test(diffusion = diffusion, model = model, test_loader = train_loader, device = device, result_path = args.result_path)
    # evaluate_diffusion_sample_ddpm(diffusion = diffusion, model = model, result_path = args.result_path)
    # evaluate_diffusion_sample_ddim(diffusion = diffusion, model = model, result_path = args.result_path)

    evaluate_diffusion_calculate_score(diffusion = diffusion, model = model, test_loader = test_loader, all_images_path=args.all_images_path, real_images_path=args.real_images_path)
    # calculate_score(all_images_path = '/home/xjy/experiment/CIFAR10/video1.0/step4/result2/sample_images_ddim/sample_images.npy', real_images_path = '/home/xjy/RealImages/cifar10.npy')
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
from tool.tool import load_diff_checkpoint, setup_seed, print_args
from tool.load_dataset_snn import load_dataset, image_to_step, image_to_event, event_normalize
from config.config import get_network_config, get_diffusion_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--epochs', default=400, type=int)
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--lr', default=2e-4, help='target learning rate')
parser.add_argument('--print_freq', default=40, type=int)
parser.add_argument('--workers', default=12, type=int, metavar='N',help='number of data loading workers (default: 10)')
parser.add_argument('--dataset_name', default='CIFAR10', type=str)
parser.add_argument('--data_path', default='/home/xjy/datasets', type=str)
parser.add_argument('--result_path', default='./result', type=str)
parser.add_argument('--model_checkpoint', default=None, type=str,action='store', dest='model_checkpoint',help='The path of checkpoint, if use checkpoint')
# Common
parser.add_argument('--img_size', default=[10,3,32,32], help='image size [B, C, H, W]')
parser.add_argument('--step',type=int,default=8)
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

def train_diffusion(diffusion, model, optimizer, mse, train_loader, test_loader, device, result_path, epochs, print_freq):  
    loss_mse_min_train = 10.0e10
    for epoch in range(epochs):
        print(f'Epoch {epoch}. Current time: {datetime.datetime.now()}.')
        # Train
        model.train()
        loss_mse_sum = 0.0
        length = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.float().to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            functional.reset_net(model)
            length += images.shape[0]
            loss_mse_sum += loss * images.shape[0]
            if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
                print("[{}/{}][{}/{}]: loss {:.4f}".format(epoch,args.epochs,i,len(train_loader),loss.item()))
        loss_mse = loss_mse_sum / length
        print("loss_mse {:.4f}".format(loss_mse))
        if epoch == 0 or loss_mse_min_train > loss_mse:
            loss_mse_min_train = loss_mse
            print("Update train model!!!!!!!!!!!!")
            torch.save(model.state_dict(), result_path+'/ckp_best.pth')
        # Save
        torch.save(model.state_dict(), result_path+'/ckp_latest.pth')

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    mse = nn.MSELoss()

    diffusion = SpikingDiffusion(noise_steps = args.noise_steps, beta_start = args.beta_start, beta_end = args.beta_end, img_size = args.img_size, device = device)
    
    train_diffusion(diffusion = diffusion, model = model, optimizer = optimizer, mse = mse, train_loader = train_loader, test_loader = test_loader, device = device, result_path = args.result_path, 
                    epochs = args.epochs, print_freq = args.print_freq)

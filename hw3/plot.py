import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import argparse
import pickle
import skimage.io
import skimage
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torchvision
from models import *
from utils import *

def plot_gan(OUT_DIR):
    torch.manual_seed(5555)
    #fig2_2
    with open("saves/gan/D_loss.pkl", "rb") as fp:   # Unpickling
        D_loss = pickle.load(fp)
    with open("saves/gan/G_loss.pkl", "rb") as fp:   # Unpickling
        G_loss = pickle.load(fp)
    with open("saves/gan/D_real_acc.pkl", "rb") as fp:   # Unpickling
        D_real_acc = pickle.load(fp)
    with open("saves/gan/D_fake_acc.pkl", "rb") as fp:   # Unpickling
        D_fake_acc = pickle.load(fp)

    #fig1_2
    rand_inputs = Variable(torch.randn(32, 100, 1, 1),volatile=True)
    G = Generator()
    G.load_state_dict(torch.load('saves/save_models/Generator.pth',map_location=lambda storage, loc: storage))
    if torch.cuda.is_available():
        rand_inputs = rand_inputs.cuda()
        G.cuda()
    G.eval()
    rand_outputs = G(rand_inputs)
    filename = os.path.join(OUT_DIR, 'fig1_2.jpg')
    torchvision.utils.save_image(rand_outputs.cpu().data, filename, nrow=8)

def plot_acgan(OUT_DIR):
    torch.manual_seed(2222)

    #fig3_2
    with open("saves/acgan/D_loss.pkl", "rb") as fp:   # Unpickling
        D_loss = pickle.load(fp)
    with open("saves/acgan/G_loss.pkl", "rb") as fp:   # Unpickling
        G_loss = pickle.load(fp)
    with open("saves/acgan/D_real_acc.pkl", "rb") as fp:   # Unpickling
        D_real_acc = pickle.load(fp)
    with open("saves/acgan/D_fake_acc.pkl", "rb") as fp:   # Unpickling
        D_fake_acc = pickle.load(fp)
    with open("saves/acgan/Real_attr_loss.pkl", "rb") as fp:   # Unpickling
        Real_attr_loss = pickle.load(fp)
    with open("saves/acgan/Fake_attr_loss.pkl", "rb") as fp:   # Unpickling
        Fake_attr_loss = pickle.load(fp)
    

#fig2_2
    fixed_noise = torch.randn((10, 100, 1, 1))
    fixed_noise = torch.cat((fixed_noise,fixed_noise), dim=0)
    fixed_label = torch.cat((torch.zeros((10,1,1,1)), torch.ones((10,1,1,1))), dim=0)
    fixed_noise = torch.cat((fixed_noise,fixed_label), dim=1)
    fixed_noise = to_var(fixed_noise)
    G = ACGenerator()
    G.load_state_dict(torch.load('saves/save_models/ACGenerator.pth',map_location=lambda storage, loc: storage))
    if torch.cuda.is_available():
        G.cuda()
    G.eval()
    rand_outputs = G(fixed_noise)
    filename = os.path.join(OUT_DIR, 'fig2_2.jpg')
    torchvision.utils.save_image(rand_outputs.cpu().data, filename, nrow=10)


def main(args):
    OUT_DIR = args.out_path
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if args.mode=='gan':
        plot_gan(OUT_DIR)
    if args.mode=='acgan':
        plot_acgan(OUT_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW2 plot figure')
    parser.add_argument('--out_path', help='output figure directory', type=str)
    parser.add_argument('--mode', help='output figure directory', type=str)
    args = parser.parse_args()
    main(args)
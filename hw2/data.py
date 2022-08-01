import os
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image


class SegDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.root_dir = os.path.join(args.data_path, mode)
        if args.test_path != '' and mode == "test":
            self.img_dir = os.path.join(args.test_path, 'img')
            self.split_dir = os.path.abspath(os.path.join(self.img_dir, os.pardir))
            self.seg_dir = os.path.join(self.split_dir, "seg")
            self.path_lists = [ (os.path.join(self.img_dir,img), os.path.join(self.seg_dir,img)) for img in os.listdir(self.img_dir) ]
        else:
        	self.img_dir = os.path.join(self.root_dir, "img")
        	self.seg_dir = os.path.join(self.root_dir, "seg")
        	self.path_lists = [ (os.path.join(self.img_dir,img), os.path.join(self.seg_dir,img)) for img in os.listdir(self.img_dir) ]
        if self.mode == 'train':
            self.transforms = transforms.Compose([
                transforms.ColorJitter(),
                transforms.ToTensor(), # (H,W,C)->(C,H,W) [0,255]->[0,1.0]
                transforms.Normalize(args.img_mean, args.img_std)
            ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(args.img_mean, args.img_std)
            ])
        self.transforms_seg = transforms.ToTensor()

    def __len__(self):
        return len(self.path_lists)

    def __getitem__(self, index):
        img_name = self.path_lists[index][0]
        seg_name = self.path_lists[index][1]
        ''' read image '''
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        ''' read segmentation '''
        seg = Image.open(seg_name)
        seg = (self.transforms_seg(seg)*255).squeeze().long()
        return img_name.split('/')[-1], img , seg

        

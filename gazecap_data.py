import numpy as np 
import random
import json
import os
import shutil
from glob import glob
from PIL import Image
import torch
import scipy.io as sio
from torchvision.transforms import Normalize, Resize, Compose, ToTensor
import sys
from torch.utils.data import Dataset, DataLoader

class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = ToTensor()(meanImg)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """       
        return tensor.sub(self.meanImg)


class gazecap_data(Dataset):
    def __init__(self, root, phase='train', size=(224,224), grid_size=(25,25)):
        self.faceMean = sio.loadmat('./Metadata/mean_face_224.mat', squeeze_me=True, struct_as_record=False)['image_mean']/255.
        self.eyeLeftMean = sio.loadmat('./Metadata/mean_left_224.mat', squeeze_me=True, struct_as_record=False)['image_mean']/255.
        self.eyeRightMean = sio.loadmat('./Metadata/mean_right_224.mat', squeeze_me=True, struct_as_record=False)['image_mean']/255.
        
        self.root = root
        self.gridSize = grid_size
        if(phase=='test'):
            self.files = glob(root+"*.jpg")
        else:
            self.files = glob(root+"/images/*.jpg")
            
        self.phase = phase
        self.size = size
        self.transformFace = self.get_transforms(self.phase, self.size, self.faceMean)
        self.transformEyeL = self.get_transforms(self.phase, self.size, self.eyeLeftMean)
        self.transformEyeR = self.get_transforms(self.phase, self.size, self.eyeRightMean)
        
    def get_transforms(self, phase, size, meanImg):
        list_transforms = []
        list_transforms.extend(
            [
                Resize((size[0],size[1])),
                ToTensor(),
                SubtractMean(meanImg=meanImg),
            ]
        )
        list_trfms = Compose(list_transforms)
        return list_trfms
    
    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen,], np.float32)
        
        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2]) 
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3]) 
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, idx):
        fname = self.files[idx]
        image = Image.open(self.files[idx])
        meta = json.load(open(fname.replace('images', 'meta').replace('.jpg', '.json')))
        
        
        fx, fy, fw, fh = meta['face_x'], meta['face_y'], meta['face_w'], meta['face_h']
        lx, ly, lw, lh = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        rx, ry, rw, rh = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        imFace = image.crop((max(0, fx), max(0, fy), max(0, fx+fw), max(0, fy+fh)))
        imEyeL = image.crop((max(0, lx), max(0, ly), max(0, lx+lw), max(0, ly+lh)))
        imEyeR = image.crop((max(0, rx), max(0, ry), max(0, rx+rw), max(0, ry+rh)))
        
        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)

        gaze = np.array([meta['dot_xcam'], meta['dot_y_cam']], np.float32)

        faceGrid = self.makeGrid(meta['face_grid'])

        return imFace, imEyeL, imEyeR, faceGrid, gaze

    def __len__(self):
        return len(self.files)
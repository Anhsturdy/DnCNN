import os
import glob
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
import cv2
from utils import data_augmentation

def normalize(data):
    """Normalize image data to range [0, 1]."""
    return data / 255.0

def Im2Patch(img, win, stride=1):
    """Extracts image patches for training."""
    k = 0
    endc, endw, endh = img.shape
    patch = img[:, 0:endw-win+1:stride, 0:endh-win+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            Y[:, k, :] = patch.reshape(endc, TotalPatNum)
            k += 1
    
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(data_path, patch_size, stride, aug_times=1):
    """Prepares training and validation datasets."""
    print('Processing training data...')
    scales = [1, 0.9, 0.8, 0.7]
    train_files = sorted(glob.glob(os.path.join(data_path, 'train', '*.png')))
    
    with h5py.File('train.h5', 'w') as h5f:
        train_num = 0
        for file in train_files:
            img = cv2.imread(file)
            h, w, c = img.shape

            for scale in scales:
                new_h, new_w = int(h * scale), int(w * scale)
                Img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)  # Fix h, w swap
                Img = np.expand_dims(Img[:, :, 0], 0).astype(np.float32)
                Img = normalize(Img)
                patches = Im2Patch(Img, win=patch_size, stride=stride)

                print(f"File: {file}, Scale: {scale:.1f}, #Samples: {patches.shape[3] * aug_times}")

                for n in range(patches.shape[3]):
                    data = patches[:, :, :, n].copy()
                    h5f.create_dataset(str(train_num), data=data)
                    train_num += 1

                    for m in range(aug_times - 1):
                        data_aug = data_augmentation(data, np.random.randint(1, 8))  # Fixed mode range
                        h5f.create_dataset(f"{train_num}_aug_{m+1}", data=data_aug)
                        train_num += 1

    print('Processing validation data...')
    val_files = sorted(glob.glob(os.path.join(data_path, 'Set12', '*.png')))

    with h5py.File('val.h5', 'w') as h5f:
        val_num = 0
        for file in val_files:
            print(f"File: {file}")
            img = cv2.imread(file)
            img = np.expand_dims(img[:, :, 0], 0).astype(np.float32)
            img = normalize(img)
            h5f.create_dataset(str(val_num), data=img)
            val_num += 1

    print(f'Training set samples: {train_num}')
    print(f'Validation set samples: {val_num}')

class Dataset(udata.Dataset):
    """Custom dataset class for training and validation."""
    def __init__(self, train=True):
        self.train = train
        self.h5_file = 'train.h5' if train else 'val.h5'
        
        with h5py.File(self.h5_file, 'r') as h5f:
            self.keys = list(h5f.keys())
        
        random.shuffle(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.h5_file, 'r') as h5f:
            key = self.keys[index]
            data = np.array(h5f[key])

        return torch.tensor(data, dtype=torch.float32)  # Ensures PyTorch-compatible dtype

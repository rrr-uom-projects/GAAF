import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from scipy.stats import norm
from scipy.ndimage import distance_transform_edt as dist_xfm
import random
from os.path import join

from .utils import getFiles

class Locator_Dataset(data.Dataset):
    def __init__(self, imagedir, image_inds, CoM_targets, shift_augment=True, flip_augment=True):
        self.imagedir = imagedir
        self.availableImages = [sorted(getFiles(imagedir))[ind] for ind in image_inds]
        self.targets = np.array([CoM_targets[image_fname.replace('.npy','')] for image_fname in self.availableImages])
        self.shifts = shift_augment
        self.flips = flip_augment
        self.gaussDist = norm(scale=10)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        imageToUse = self.availableImages[idx]
        ct_im = np.load(join(self.imagedir, imageToUse))
        ## expecting ct_im to be 4D (3ch, cc, ap, lr)
        ct_spatial_size = ct_im.shape[1:]
        coord_target = self.targets[idx].copy()
        # Augmentations
        if self.shifts or self.flips:
            if self.shifts:
                # custom shifting function using padding and cropping
                # since we ony want to shift on the grid this approach is mad quicker than scipy.ndimage.shift (especially in 4D)
                # find shift values
                max_cc_shift, max_ap_shift, max_lr_shift = 2, 4, 4
                cc_shift, ap_shift, lr_shift = random.randint(-max_cc_shift, max_cc_shift), random.randint(-max_ap_shift, max_ap_shift), random.randint(-max_lr_shift, max_lr_shift)
                # pad for shifting into
                pad_width = ((0,0), (max_cc_shift, max_cc_shift), (max_ap_shift, max_ap_shift), (max_lr_shift, max_lr_shift))
                ct_im = np.pad(ct_im, pad_width=pad_width, mode='constant')
                # crop to complete shift - trust this insanity
                ct_im = ct_im[:, max_cc_shift+cc_shift:ct_spatial_size[0]+max_cc_shift+cc_shift, max_ap_shift+ap_shift:ct_spatial_size[1]+max_ap_shift+ap_shift, max_lr_shift+lr_shift:ct_spatial_size[2]+max_lr_shift+lr_shift]
                # nudge the target to match the shift
                coord_target[0] -= cc_shift
                coord_target[1] -= ap_shift
                coord_target[2] -= lr_shift
            if self.flips:
                if random.choice([True, False]):
                    # implement LR flip
                    ct_im = np.flip(ct_im, axis=3).copy()
                    coord_target[2] = ct_spatial_size[2] - coord_target[2]

        # now convert target to heatmap target
        # use new off grid heatmap generation
        t = np.indices(dimensions=ct_spatial_size).astype(float)
        dist_map = np.sqrt(np.sum([np.power((2*(t[0] - coord_target[0])), 2), np.power((t[1] - coord_target[1]), 2), np.power((t[2] - coord_target[2]), 2)], axis=0))
        h_target = self.gaussDist.pdf(dist_map)
        h_target *= (1 / np.max(h_target))
        return {'ct_im': ct_im, 'target': coord_target, 'h_target': h_target[np.newaxis]} # added channels axis here

    def __len__(self):
        return len(self.availableImages)

class Locator_Testset(data.Dataset):
    def __init__(self, imagedir, image_inds, CoM_targets):
        self.imagedir = imagedir
        self.availableImages = [sorted(getFiles(imagedir))[ind] for ind in image_inds]
        self.targets = np.array([CoM_targets[image_fname.replace('.npy','')] for image_fname in self.availableImages])
        self.gaussDist = norm(scale=10)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        imageToUse = self.availableImages[idx]
        ct_im = np.load(join(self.imagedir, imageToUse))
        ## expecting ct_im to be 4D (3ch, cc, ap, lr)
        ct_spatial_size = ct_im.shape[1:]
        coord_target = self.targets[idx].copy()
        
        # now convert target to heatmap target
        # use new off grid heatmap generation
        t = np.indices(dimensions=ct_spatial_size).astype(float)
        dist_map = np.sqrt(np.sum([np.power((2*(t[0] - coord_target[0])), 2), np.power((t[1] - coord_target[1]), 2), np.power((t[2] - coord_target[2]), 2)], axis=0))
        h_target = self.gaussDist.pdf(dist_map)
        h_target *= (1 / np.max(h_target))
        return {'ct_im': ct_im, 'target': coord_target, 'h_target': h_target[np.newaxis], "fname": imageToUse} # added channels axis here

    def __len__(self):
        return len(self.availableImages)
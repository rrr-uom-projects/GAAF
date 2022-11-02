# train_locator.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import os
from os.path import join
import argparse as ap
import pickle
from tqdm import tqdm

from .model import Locator, Attention_Locator
from .trainers import Locator_trainer
from .dataset import Locator_Dataset, Locator_Testset
from .utils import *

class train_locator():
    def __init__(self, args):
        # decide checkpoint directory
        try_mkdir(args.model_dir)
        self.checkpoint_dir = join(args.model_dir, "fold"+str(args.fold_num))
        
        # Create main logger
        self.logger = get_logger('Locator_Training')

        # Create the model
        if args.use_attention:
            self.model = Attention_Locator(n_targets=1, in_channels=1)
        else:
            self.model = Locator(filter_factor=1, n_targets=1, in_channels=1)
        for param in self.model.parameters():
            param.requires_grad = True

        # put the model on GPU(s)
        device='cuda'
        self.model.to(device)

        # Log the number of learnable parameters
        self.logger.info(f'Number of learnable params {get_number_of_learnable_parameters(self.model)}')
        
        train_BS = args.train_BS
        val_BS = args.val_BS
        train_workers = args.train_workers
        val_workers = args.val_workers

        # allocate ims to train, val and test
        dataset_size = len(getFiles(args.image_dir))
        train_inds, val_inds, _ = k_fold_split_train_val_test(dataset_size, fold_num=args.fold_num, seed=args.seed)

        # load in the CoM targets
        with open(join(args.CoM_targets_dir, "CoM_targets.pkl"), 'rb') as f:
            CoM_targets = pickle.load(f)

        # Create them dataloaders
        train_data = Locator_Dataset(imagedir=args.image_dir, CoM_targets=CoM_targets, image_inds=train_inds, shift_augment=True, flip_augment=True)
        train_loader = DataLoader(dataset=train_data, batch_size=train_BS, shuffle=True, pin_memory=True, num_workers=train_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
        val_data = Locator_Dataset(imagedir=args.image_dir, CoM_targets=CoM_targets, image_inds=val_inds, shift_augment=True, flip_augment=True)
        val_loader = DataLoader(dataset=val_data, batch_size=val_BS, shuffle=True, pin_memory=True, num_workers=val_workers, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))

        # Create the optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.init_lr)

        # Create learning rate adjustment strategy
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=175, verbose=True)
    
        # Create model trainer
        self.trainer = Locator_trainer( model=self.model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, train_loader=train_loader, 
                                        val_loader=val_loader, logger=self.logger, checkpoint_dir=self.checkpoint_dir, max_num_epochs=args.max_num_epochs, 
                                        patience=args.patience, iters_to_accumulate=args.iters_to_accumulate)
    
    def run_training(self):
        # Start training
        self.trainer.fit()
        # Romeo Dunn

class test_locator():
    def __init__(self, args):
        # make the results directory
        self.res_dir = join(args.model_dir, "results/") 
        try_mkdir(self.res_dir)
        self.res_fname = join(self.res_dir, f"fold{str(args.fold_num)}.npy")
        self.checkpoint_dir = join(args.model_dir, "fold"+str(args.fold_num))
        
        # Create main logger
        self.logger = get_logger('Locator_Testing')

        # Create the model
        if args.use_attention:
            self.model = Attention_Locator(n_targets=1, in_channels=1)
        else:
            self.model = Locator(filter_factor=1, n_targets=1, in_channels=1)
        for param in self.model.parameters():
            param.requires_grad = False

        # load the weights from the best checkpoint
        self.model.load_best(self.checkpoint_dir, self.logger)

        # put the model on GPU(s)
        self.device='cuda'
        self.model.to(self.device)

        # allocate ims to train, val and test
        dataset_size = len(getFiles(args.image_dir))
        _, _, test_inds = k_fold_split_train_val_test(dataset_size, fold_num=args.fold_num, seed=args.seed)

        # load in the CoM targets
        with open(join(args.CoM_targets_dir, "CoM_targets.pkl"), 'rb') as f:
            CoM_targets = pickle.load(f)

        # load in the spacings
        with open(join(args.CoM_targets_dir, "spacings.pkl"), 'rb') as f:
            self.spacings = pickle.load(f)

        # Create them dataloaders
        test_data = Locator_Testset(imagedir=args.image_dir, CoM_targets=CoM_targets, image_inds=test_inds)
        self.test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, pin_memory=False, num_workers=0, worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)))
    
    def run_testing(self):
        # setup results array
        results = np.zeros((len(self.test_loader), 3))
        # Start testing
        for im_idx, sample in enumerate(tqdm(self.test_loader)):
            ct_im = sample['ct_im'].type(torch.FloatTensor)
            target = sample['target'].numpy()[0]
            h_target = sample['h_target'].type(torch.FloatTensor)
            fname = sample['fname'][0]
            # send tensors to GPU
            ct_im = ct_im.to(self.device)
            h_target = h_target.to(self.device)
            # get the model prediction and MSE loss
            output = self.model(ct_im)
            loss = torch.nn.MSELoss()(output, h_target)
            # get the predicted location coordinates
            pred_heatmap = output.detach().cpu().numpy()[0,0]
            im_shape = pred_heatmap.shape
            argmax_pred = np.unravel_index(np.argmax(pred_heatmap), im_shape)
            t = np.indices(im_shape).astype(np.float)
            # define 3D gauss function
            def prod_3d_gaussian(t, mu_i, mu_j, mu_k):
                pos = np.array([mu_i, mu_j, mu_k])
                t = t.reshape((3,) + im_shape)
                dist_map = np.sqrt(np.sum([np.power((2*(t[0] - pos[0])), 2), np.power((t[1] - pos[1]), 2), np.power((t[2] - pos[2]), 2)], axis=0))
                gaussian = np.array(norm(scale=10).pdf(dist_map), dtype=np.float)
                return gaussian.ravel()
            # get the predicted location coordinates
            p_opt, _ = curve_fit(prod_3d_gaussian, t.ravel(), pred_heatmap.ravel(), p0=argmax_pred)
            # find euclidean distance
            spacing = self.spacings[fname.replace('.npy','.nii')][1]
            target *= spacing
            argmax_pred *= spacing
            p_opt *= spacing
            argmax_res = np.sqrt(np.power(target[0] - argmax_pred[0], 2) + np.power(target[1] - argmax_pred[1], 2) + np.power(target[2] - argmax_pred[2], 2))
            fitted_res = np.sqrt(np.power(target[0] - p_opt[0], 2) + np.power(target[1] - p_opt[1], 2) + np.power(target[2] - p_opt[2], 2))
            # store results
            results[im_idx, 0] = loss
            results[im_idx, 1] = argmax_res
            results[im_idx, 2] = fitted_res
        # save results array
        np.save(self.res_fname, results)
        # Romeo Dunn

def setup_argparse():
    parser = ap.ArgumentParser(prog="Main training program for 3D location-finding network \"Locator\"")
    parser.add_argument("--fold_num", choices=[1,2,3,4,5], type=int, help="The fold number for the kfold cross validation")
    parser.add_argument("--model_dir", type=str, help="The file path where the model weights will be saved to")
    parser.add_argument("--use_attention", default=True, type=lambda x:str2bool(x), help="Use model with attention gates?")
    parser.add_argument("--image_dir", type=str, help="The file path of the folder containing the ct images")
    parser.add_argument("--CoM_targets_dir", type=str, help="The path of the directory containing the CoM targets")
    parser.add_argument("--train_BS", type=int, default=1, help="The training batch size")
    parser.add_argument("--val_BS", type=int, default=1, help="The validation batch size")
    parser.add_argument("--train_workers", type=int, default=4, help="The no. of training workers")
    parser.add_argument("--val_workers", type=int, default=4, help="The no. of validation workers")
    parser.add_argument("--init_lr", type=float, default=0.005, help="The initial learning rate for the Adam optimiser")
    parser.add_argument("--max_num_epochs", type=int, default=1000, help="The maximum number of epochs to train for")
    parser.add_argument("--patience", type=int, default=500, help="Training will stop this many epochs after after the last improvement of the val_loss (or max_num_epochs)")
    parser.add_argument("--iters_to_accumulate", type=int, default=1, help="Gradient accumulation for simulating larger batches")
    parser.add_argument("--seed", type=int, default=100457, help="The seed to shuffle the images for the train, val and test sets")
    args = parser.parse_args()
    return args
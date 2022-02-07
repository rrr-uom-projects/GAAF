# train_locator.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import os
from os.path import join
import argparse as ap
import pickle

from training.model import Locator, Attention_Locator
from training.trainers import Locator_trainer
from training.dataset import Locator_Dataset
from utils.utils import *

class train_locator():
    def __init__(self, args):
        # decide checkpoint directory
        try_mkdir(args.checkpoint_dir)
        self.checkpoint_dir = join(args.checkpoint_dir, "fold"+str(args.fold_num))
        
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
        with open(args.CoM_targets, 'rb') as f:
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


def setup_argparse():
    parser = ap.ArgumentParser(prog="Main training program for 3D location-finding network \"Locator\"")
    parser.add_argument("--fold_num", choices=[1,2,3,4,5], type=int, help="The fold number for the kfold cross validation")
    parser.add_argument("--checkpoint_dir", type=str, help="The file path where the model weights will be saved to")
    parser.add_argument("--use_attention", default=True, type=lambda x:str2bool(x), help="Use model with attention gates?")
    parser.add_argument("--train_BS", type=int, default=1, help="The training batch size")
    parser.add_argument("--val_BS", type=int, default=1, help="The validation batch size")
    parser.add_argument("--train_workers", type=int, default=4, help="The no. of training workers")
    parser.add_argument("--val_workers", type=int, default=4, help="The no. of validation workers")
    parser.add_argument("--image_dir", type=str, help="The file path of the folder containing the ct images")
    parser.add_argument("--CoM_targets", type=str, help="The file path of the file containing the CoM targets")
    parser.add_argument("--init_lr", type=float, default=0.005, help="The initial learning rate for the Adam optimiser")
    parser.add_argument("--max_num_epochs", type=int, default=1000, help="The maximum number of epochs to train for")
    parser.add_argument("--patience", type=int, default=500, help="Training will stop this many epochs after after the last improvement of the val_loss (or max_num_epochs)")
    parser.add_argument("--iters_to_accumulate", type=int, default=1, help="Gradient accumulation for simulating larger batches")
    parser.add_argument("--seed", type=int, default=100457, help="The seed to shuffle the images for the train, val and test sets")
    args = parser.parse_args()
    return args
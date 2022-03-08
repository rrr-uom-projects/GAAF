from os.path import join
import argparse as ap
from tqdm import tqdm

import torch
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
from scipy.optimize import curve_fit
from scipy.stats import norm

from .model import Locator, Attention_Locator 
from .utils import *

class Locator_inference_module:
    def __init__(self, args, test=False):
        if test:
            return
        # setup paths
        self.model_dir = args.model_dir
        self.in_image_dir = args.in_image_dir
        self.out_image_dir = args.out_image_dir
        try_mkdir(self.out_image_dir)
        # determine if masks should be involved
        self.masks = False
        if (args.in_mask_dir != "None") != bool(args.out_mask_dir != "None"): # XOR
            raise ValueError("Make sure to provide both the input and output directories for your masks...")
        elif args.in_mask_dir != "None":
            self.masks = True
            self.in_mask_dir = args.in_mask_dir
            self.out_mask_dir = args.out_mask_dir
            try_mkdir(self.out_mask_dir)
        # setup model
        self.device = 'cuda'
        self.setup_model(args)
        # save in/out resolution settings
        self.Locator_image_resolution = tuple([int(res) for res in args.Locator_image_resolution])
        self.cropped_image_resolution = tuple([int(res) for res in args.cropped_image_resolution])
        self._check_resolutions()
        # read in image fnames to run inference over
        self.pat_fnames = sorted(getFiles(self.in_image_dir))
        if self.masks:
            self.mask_fnames = sorted(getFiles(self.in_mask_dir))
        self._check_fnames()
        # determine if coords output is also desired
        self.store_coords = False
        if args.out_CoMs_dir != "None":
            self.store_coords = True
            self.out_CoMs_dir = args.out_CoMs_dir
            try_mkdir(self.out_CoMs_dir)
            self.coords = {}
    
    def setup_model(self, args):
        if args.use_attention:
            self.model = Attention_Locator(n_targets=1, in_channels=1)
        else:
            self.model = Locator(filter_factor=1, n_targets=1, in_channels=1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)
        self.model.load_best(self.model_dir, logger=None)
        self.model.eval()

    def run_inference(self):
        if self.masks:
            self.run_inference_with_masks()
        else:
            self.run_inference_no_masks()

    def run_inference_no_masks(self):
        # perform inference for all images in directory            
        for pat_idx, pat_fname in enumerate(tqdm(self.pat_fnames)):
            # carries out the full Locator inference and cropping process on a single CT image
            self.nii_im = sitk.ReadImage(join(self.in_image_dir, pat_fname))
            self.spacing = self.nii_im.GetSpacing()

            # Check im direction etc. here and if flip or rot required
            if(np.sign(self.nii_im.GetDirection()[-1]) == 1):
                self.nii_im = sitk.Flip(self.nii_im, [False, False, True])

            # convert to numpy
            im = sitk.GetArrayFromImage(self.nii_im)
            if self._check_im(min_val=im.min()):
                im += 1024
            im = np.clip(im, 0, 3024)

            # resample to desired size for Locator
            init_shape = np.array(im.shape)
            im = resize(im, output_shape=self.Locator_image_resolution, order=3, preserve_range=True, anti_aliasing=True)
            final_shape = np.array(im.shape)
            resize_performed = final_shape / init_shape

            # finally perform window and level contrast preprocessing on CT -> make this a tuneable feature in future
            # NOTE: Images are expected in WM mode -> use level = HU + 1024
            # ensure to add extra axes for batch and channels
            im = windowLevelNormalize(im, level=1064, window=1600)[np.newaxis, np.newaxis]
            
            # perform inference
            raw_coords = self.inference(im)
            self.rescaled_coords = raw_coords / resize_performed
            
            # now either do cropping and save output or simply return coords
            if self.store_coords:
                # store coords in dictionary
                self.coords[pat_fname.replace('.nii','')] = self.rescaled_coords
            
            # perform cropping around located CoM and save result
            self._apply_crop()
            sitk.WriteImage(self.nii_im, join(self.out_image_dir, pat_fname))

        if self.store_coords:
            # save all coords
            self._save_coords()

    def run_inference_with_masks(self):
        # perform inference for all images in directory            
        for pat_idx, pat_fname in enumerate(tqdm(self.pat_fnames)):
            # carries out the full Locator inference and cropping process on a single CT image
            self.nii_im = sitk.ReadImage(join(self.in_image_dir, pat_fname))
            self.nii_mask = sitk.ReadImage(join(self.in_mask_dir, pat_fname))
            self.spacing = self.nii_im.GetSpacing()

            # Check im direction etc. here and if flip or rot required
            if(np.sign(self.nii_im.GetDirection()[-1]) == 1):
                self.nii_im = sitk.Flip(self.nii_im, [False, False, True])
                self.nii_mask = sitk.Flip(self.nii_mask, [False, False, True])

            # convert to numpy
            im = sitk.GetArrayFromImage(self.nii_im)
            if self._check_im(min_val=im.min()):
                im += 1024
            im = np.clip(im, 0, 3024)
            mask = sitk.GetArrayFromImage(self.nii_mask)

            # resample to desired size for Locator
            init_shape = np.array(im.shape)
            im = resize(im, output_shape=self.Locator_image_resolution, order=3, preserve_range=True, anti_aliasing=True)
            final_shape = np.array(im.shape)
            resize_performed = final_shape / init_shape

            # finally perform window and level contrast preprocessing on CT -> make this a tuneable feature in future
            # NOTE: Images are expected in WM mode -> use level = HU + 1024
            # ensure to add extra axes for batch and channels
            im = windowLevelNormalize(im, level=1064, window=1600)[np.newaxis, np.newaxis]
            
            # perform inference
            raw_coords = self.inference(im)
            self.rescaled_coords = raw_coords / resize_performed
            
            # now either do cropping and save output or simply return coords
            if self.store_coords:
                # store coords in dictionary
                self.coords[pat_fname.replace('.nii','')] = self.rescaled_coords
            
            # perform cropping around located CoM and save result
            self._apply_crop(with_mask=True)
            sitk.WriteImage(self.nii_im, join(self.out_image_dir, pat_fname))
            sitk.WriteImage(self.nii_mask, join(self.out_mask_dir, pat_fname))

        if self.store_coords:
            # save all coords
            self._save_coords()

    def inference(self, im):
        # send image to the GPU
        im = torch.tensor(im, dtype=torch.float).to(self.device)
        # New inference using fitted gaussian, come back and better comment this black magic
        # define function to fit (generates a 3D gaussian for a given point [mu_i, mu_j, mu_k] and returns the flattened array)
        def f(t, mu_i, mu_j, mu_k):
            pos = np.array([mu_i, mu_j, mu_k])
            t = t.reshape((3,) + self.Locator_image_resolution)
            dist_map = np.sqrt(np.sum([np.power((2*(t[0] - pos[0])), 2), np.power((t[1] - pos[1]), 2), np.power((t[2] - pos[2]), 2)], axis=0))
            gaussian = np.array(norm(scale=10).pdf(dist_map), dtype=np.float)
            return gaussian.ravel()
        # run model forward to generate heatmap prediction
        model_output = self.model(im).detach().cpu().numpy()[0]
        # get starting point for curve-fitting (argmax)
        argmax_pred = np.unravel_index(np.argmax(model_output), self.Locator_image_resolution)
        # do gaussian fitting
        t = np.indices(self.Locator_image_resolution).astype(np.float)
        p_opt, _ = curve_fit(f, t.ravel(), model_output.ravel(), p0=argmax_pred)
        return p_opt
    
    def _save_coords(self):
        with open(join(self.out_CoMs_dir, "coords.pkl"), 'wb') as f:
            pickle.dump(self.coords, f)

    def _apply_crop(self, with_mask=False):
        # crop the original CT down based upon the Locator CoM coords prediction
        buffers = np.array(self.cropped_image_resolution) // 2
        # Added error case where CoM too close to image boundary -> clip CoM coord to shift sub-volume inside image volume 
        orig_im_shape = np.array(self.nii_im.GetSize())[[2,1,0]]
        self.rescaled_coords = np.clip(self.rescaled_coords, a_min=buffers, a_max=orig_im_shape-buffers)
        # determine crop boundaries
        low_crop, hi_crop = self.rescaled_coords - buffers, self.rescaled_coords + buffers
        # cast to int for index slice cropping
        low_crop, hi_crop = np.round(low_crop).astype(int), np.round(hi_crop).astype(int)
        # slice original image to crop - NOTE: nifti image indexing order is lr, ap, cc
        self.nii_im = self.nii_im[low_crop[2]:hi_crop[2], low_crop[1]:hi_crop[1], low_crop[0]:hi_crop[0]]
        if with_mask:
            self.nii_mask = self.nii_mask[low_crop[2]:hi_crop[2], low_crop[1]:hi_crop[1], low_crop[0]:hi_crop[0]]
    
    def _check_resolutions(self):
        if not isinstance(self.Locator_image_resolution, tuple) or len(self.Locator_image_resolution) != 3:
            raise ValueError("Locator_image_resolution argument must be 3 space-separated integers -> cc, ap, lr voxels")
        if not isinstance(self.cropped_image_resolution, tuple) or len(self.cropped_image_resolution) != 3:
            raise ValueError("cropped_image_resolution argument must be 3 space-separated integers -> cc, ap, lr voxels")

    def _check_fnames(self):
        for pat_fname in self.pat_fnames:
            if ".nii" not in pat_fname:
                raise NotImplementedError(f"Sorry! Inference is currently only written for nifti (.nii) images...\n found: {pat_fname} in --in_image_dir")
        if self.masks:
            # First check all masks in directory are niftis
            for mask_fname in self.mask_fnames:
                if ".nii" not in mask_fname:
                    raise NotImplementedError(f"Sorry! Inference is currently only written for nifti (.nii) images...\n found: {mask_fname} in --in_mask_dir")
            # now check that all images and masks are in matching pairs
            for pat_fname in self.pat_fnames:
                if pat_fname not in self.mask_fnames:
                    raise ValueError(f"Whoops, it looks like your images and masks aren't in matching pairs!\n found: {pat_fname} in the image directory, but not in the mask directory...")
            for mask_fname in self.mask_fnames:
                if mask_fname not in self.pat_fnames:
                    raise ValueError(f"Whoops, it looks like your images and masks aren't in matching pairs!\n found: {pat_fname} in the mask directory, but not in the image directory...")
    
    def _check_im(self, min_val):
        if min_val < 0:
            print(f"Expected CT in WM mode (min intensity at 0), instead fname: {pat_fname} min at {im.min()} -> adjusting...")
            return True
        return False


def setup_argparse(test_args=None):
    parser = ap.ArgumentParser(prog="Main inference program for 3D location-finding network \"Locator\"")
    parser.add_argument("--model_dir", type=str, help="The file path where the model weights are saved to", required=True)
    parser.add_argument("--use_attention", default=True, type=lambda x:str2bool(x), help="Doe the model with attention gates?")
    parser.add_argument("--in_image_dir", type=str, help="The file path of the folder containing the input CT images", required=True)
    parser.add_argument("--out_image_dir", type=str, help="The file path where the cropped CT subvolumes will be saved to", required=True)
    parser.add_argument("--out_CoMs_dir", type=str, default="None", help="The file path where the resultant CoM coordinates will be saved to")
    parser.add_argument("--in_mask_dir", type=str, help="The file path of the folder containing the input masks")
    parser.add_argument("--out_mask_dir", type=str, help="The file path where the cropped mask subvolumes will be saved to")
    parser.add_argument("--Locator_image_resolution", nargs="+", default=[64,256,256], help="Image resolution for Locator, pass in cc, ap, lr order")
    parser.add_argument("--cropped_image_resolution", nargs="+", default=[64,128,128], help="The size of the output crop desired (around identified CoM)")
    args = parser.parse_args(test_args) # if test args==None then parse_args will fall back on sys.argv
    return args

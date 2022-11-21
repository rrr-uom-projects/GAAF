from os.path import join
import argparse as ap
from tqdm import tqdm

import torch
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize, rescale
from scipy.optimize import curve_fit
from scipy.stats import norm

from .model import Locator, Attention_Locator 
from .utils import *

class Locator_inference_module:
    def __init__(self, args, test=False):
        if test:
            self.output_crop = True
            return
        # setup paths
        self.model_dir = args.model_dir
        self.in_image_dir = args.in_image_dir
        # setup model
        self.device = 'cuda'
        self.setup_model(args)
        # save in/out resolution settings
        self.Locator_image_resolution = tuple([int(res) for res in args.Locator_image_resolution])
        # read in image fnames to run inference over
        self.pat_fnames = sorted(getFiles(self.in_image_dir))
        # Determine what sort of output is desired
        # Cropped images (and masks)?
        self.output_crop = args.output_crop
        if self.output_crop:
            self.cropped_image_resolution = tuple([int(res) for res in args.cropped_image_resolution])
            self.out_image_dir = args.out_image_dir
            try_mkdir(self.out_image_dir)
            # determine if masks should be involved
            self.masks = False
            if (args.in_mask_dir != None) != bool(args.out_mask_dir != None): # XOR
                raise ValueError("Make sure to provide both the input and output directories for your masks...")
            elif args.in_mask_dir != None:
                self.masks = True
                self.in_mask_dir = args.in_mask_dir
                self.mask_fnames = sorted(getFiles(self.in_mask_dir))
                self._check_fnames()
                self.out_mask_dir = args.out_mask_dir
                try_mkdir(self.out_mask_dir)
            # set flag to determine axial resizing of images is required
            self.scale_slice_thickness = False
            if args.cropped_image_slice_thickness != -23.5:
                self.scale_slice_thickness = True
                self.desired_slice_thickness = args.cropped_image_slice_thickness
        self._check_resolutions()
        # is coords output is desired?
        self.coords = {}
        self.output_coords = args.output_coords
        if self.output_coords:
            self.out_CoMs_dir = args.out_CoMs_dir
            try_mkdir(self.out_CoMs_dir)
    
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
        # perform inference for all images in directory            
        for pat_idx, pat_fname in enumerate(tqdm(self.pat_fnames)):
            # carries out the full Locator inference and cropping process on a single CT image
            self.nii_im = sitk.ReadImage(join(self.in_image_dir, pat_fname))
            if self.masks:
                self.nii_mask = sitk.ReadImage(join(self.in_mask_dir, pat_fname))
            self.spacing = self.nii_im.GetSpacing()

            # Check im direction etc. here and if flip or rot required
            flipped = False
            if(np.sign(self.nii_im.GetDirection()[-1]) == 1):
                flipped = True
                self.nii_im = sitk.Flip(self.nii_im, [False, False, True])
                if self.masks:
                    self.nii_mask = sitk.Flip(self.nii_mask, [False, False, True])

            # convert to numpy
            im = sitk.GetArrayFromImage(self.nii_im)
            if self._check_im(mean_val=im.mean(), fname=pat_fname):
                im += 1024
            im = np.clip(im, 0, 3024)
            if self.masks:
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
       
            # resize original images to desired slice thickness prior to cropping
            if self.scale_slice_thickness:
                self._resample_slice_thickness_ct()
                if self.masks:
                    self._resample_slice_thickness_mask()
                self._check_resample()
            
            # now either do cropping and save output or simply return coords
            r_c = self.rescaled_coords.copy()
            # flip the coords back if image and mask were originally flipped
            if flipped:
                r_c[0] = self.Locator_image_resolution[0] - r_c[0]
            # store coords in dictionary
            self.coords[pat_fname.replace('.nii','')] = r_c

            # perform cropping around located CoM and save result
            if self.output_crop:
                if self._apply_crop(pat_fname=pat_fname):
                    self._check_output_size()
                    sitk.WriteImage(self.nii_im, join(self.out_image_dir, pat_fname))
                    if self.masks:
                        sitk.WriteImage(self.nii_mask, join(self.out_mask_dir, pat_fname))

        if self.output_coords:
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

    def _resample_slice_thickness_ct(self):
        spacing = self.nii_im.GetSpacing()
        # is resampling necessary?
        if spacing[2] > self.desired_slice_thickness - 0.1 and spacing[2] < self.desired_slice_thickness + 0.1:
            # slice thickness is almost correct already - no need to resample
            return
        # resample image to desired slice thickness
        direction = self.nii_im.GetDirection()
        origin = self.nii_im.GetOrigin()
        pixel_grid = sitk.GetArrayFromImage(self.nii_im)
        # determine scale factor
        scale_factor = spacing[2] / self.desired_slice_thickness
        pixel_grid = rescale(pixel_grid, [scale_factor, 1, 1], order=3, preserve_range=True, anti_aliasing=True)
        # put it back
        self.nii_im = sitk.GetImageFromArray(pixel_grid)
        self.nii_im.SetDirection(direction)
        self.nii_im.SetOrigin(origin)
        self.nii_im.SetSpacing([spacing[0], spacing[1], self.desired_slice_thickness])
        # scale coords accordingly
        self.rescaled_coords[0] *= scale_factor
    
    def _resample_slice_thickness_mask(self):
        spacing = self.nii_mask.GetSpacing()
        # is resampling necessary? 
        if spacing[2] > self.desired_slice_thickness - 0.1 and spacing[2] < self.desired_slice_thickness + 0.1:
            # slice thickness is almost correct already - no need to resample
            return
        # resample image to desired slice thickness
        direction = self.nii_mask.GetDirection()
        origin = self.nii_mask.GetOrigin()
        pixel_grid = sitk.GetArrayFromImage(self.nii_mask)
        # determine scale factor
        scale_factor = spacing[2] / self.desired_slice_thickness
        pixel_grid = rescale(pixel_grid, [scale_factor, 1, 1], order=0, preserve_range=True, anti_aliasing=False)
        # put it back
        self.nii_mask = sitk.GetImageFromArray(pixel_grid)
        self.nii_mask.SetDirection(direction)
        self.nii_mask.SetOrigin(origin)
        self.nii_mask.SetSpacing([spacing[0], spacing[1], self.desired_slice_thickness])

    def _check_resample(self):
        assert(self.nii_im.GetSpacing() == self.nii_mask.GetSpacing())
        assert(self.nii_im.GetSize() == self.nii_mask.GetSize())

    def _apply_crop(self, pat_fname=None):
        # crop the original CT down based upon the Locator CoM coords prediction
        buffers = np.array(self.cropped_image_resolution) // 2
        # Added error case where CoM too close to image boundary -> clip CoM coord to shift sub-volume inside image volume 
        orig_im_shape = np.array(self.nii_im.GetSize())[[2,1,0]]
        #print(self.rescaled_coords)
        self.rescaled_coords = np.clip(self.rescaled_coords, a_min=buffers, a_max=orig_im_shape-buffers)
        #print(self.rescaled_coords)
        # determine crop boundaries
        low_crop, hi_crop = self.rescaled_coords - buffers, self.rescaled_coords + buffers
        # cast to int for index slice cropping
        low_crop, hi_crop = np.round(low_crop).astype(int), np.round(hi_crop).astype(int)
        # determine if resizing is necessary
        if orig_im_shape[0] < self.cropped_image_resolution[0]:
            if self.scale_slice_thickness:
                print(f"Scaled image resolution is smaller than cropped image resolution.\nCheck the derised slice thickness is appropriate for {pat_fname}. SKipping for now ...")
                return False
            print(f"Original image resolution is smaller than cropped image resolution. Resizing not allowed. Skipping {pat_fname}.")
            return False
        # slice original image to crop - NOTE: nifti image indexing order is lr, ap, cc
        self.nii_im = self.nii_im[low_crop[2]:hi_crop[2], low_crop[1]:hi_crop[1], low_crop[0]:hi_crop[0]]
        if self.masks:
            self.nii_mask = self.nii_mask[low_crop[2]:hi_crop[2], low_crop[1]:hi_crop[1], low_crop[0]:hi_crop[0]]
        return True
    
    def _check_output_size(self):
        # check if output image size is correct
        if (np.array(self.nii_im.GetSize())[[2,0,1]] != self.cropped_image_resolution).all():
            raise ValueError(f"Output image size {self.nii_im.GetSize()[[2,0,1]]} is not equal to cropped image size {self.cropped_image_resolution}.")
        if self.masks and (np.array(self.nii_mask.GetSize())[[2,0,1]] != self.cropped_image_resolution).all():
            raise ValueError(f"Output mask size {self.nii_mask.GetSize()[[2,0,1]]} is not equal to cropped image size {self.cropped_image_resolution}.")
        return True

    def _check_resolutions(self):
        if not isinstance(self.Locator_image_resolution, tuple) or len(self.Locator_image_resolution) != 3:
            raise ValueError("Locator_image_resolution argument must be 3 space-separated integers -> cc, ap, lr voxels")
        if self.output_crop:
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
    
    def _check_im(self, mean_val, fname):
        if mean_val < 0:
            print(f"WARNING: Image {fname} has negative values! This is not supported by the current preprocessing pipeline.\nPlease ensure that your images are in Hounsfield Units (HU) + 1024 (WM mode) and that the minimum value is >= 0")
            print(f"Expected CT in WM mode (mean intensity > 0), instead fname: {fname} mean at {mean_val} -> adjusting...")
            return True
        return False


def setup_argparse(test_args=None):
    parser = ap.ArgumentParser(prog="Main inference program for 3D location-finding network \"Locator\"")
    parser.add_argument("--model_dir", type=str, help="The file path where the model weights are saved to", required=True)
    parser.add_argument("--use_attention", default=True, type=lambda x:str2bool(x), help="Use the model with attention gates?")
    parser.add_argument("--Locator_image_resolution", nargs="+", default=[64,256,256], help="Image resolution for Locator, pass in cc, ap, lr order")
    parser.add_argument("--in_image_dir", type=str, help="The file path of the folder containing the input CT images", required=True)
    parser.add_argument("--in_mask_dir", type=str, help="The file path of the folder containing the (optional) input masks")
    parser.add_argument("--output_crop", type=lambda x:str2bool(x), help="Output the cropped image and mask?", required=True)
    parser.add_argument("--out_image_dir", type=str, help="The file path where the cropped CT subvolumes will be saved to (optional)")
    parser.add_argument("--out_mask_dir", type=str, help="The file path where the cropped mask subvolumes will be saved to (optional)")
    parser.add_argument("--cropped_image_resolution", nargs="+", default=[64,128,128], help="The size of the output crop desired (around identified CoM) (optional)")
    parser.add_argument("--output_coords", type=lambda x:str2bool(x), help="Output the coordinates?", required=True)
    parser.add_argument("--out_CoMs_dir", type=str, help="The file path where the coordinates will be saved to")
    parser.add_argument("--cropped_image_slice_thickness", type=float, default=-23.5, help="The axial slice width to rescale the output images to (mm)")
    args = parser.parse_args(test_args) # if test args==None then parse_args will fall back on sys.argv
    return args

from os.path import join
import argparse as ap
from tqdm import tqdm

import torch
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
from scipy.optimize import curve_fit
from scipy.stats import norm

from training.model import Locator, Attention_Locator 
from utils.utils import *

class Locator_inference_module:
    def __init__(self, args):
        # setup paths
        self.model_dir = args.model_dir
        self.image_dir = args.image_dir
        self.output_dir = args.output_dir
        try_mkdir(self.output_dir)
        # setup model
        self.device = 'cuda'
        self.setup_model(args)
        # save in/out resolution settings
        self.Locator_resolution = tuple([int(res) for res in args.Locator_resolution])
        self.output_resolution = tuple([int(res) for res in args.output_resolution])
        self._check_resolutions()
        # read in image fnames to run inference over
        self.pat_fnames = sorted(getFiles(self.image_dir))
        self._check_pat_fnames()
        # determine what output is desired
        self.coords_only = True if args.subvolumes_or_coords == "coords" else False
        if self.coords_only:
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

    def _check_resolutions(self):
        if not isinstance(self.Locator_resolution, tuple) or len(self.Locator_resolution) != 3:
            print("Locator_resolution argument must be a length-3 tuple -> (cc, ap, lr) voxels")
            exit()
        if not isinstance(self.output_resolution, tuple) or len(self.output_resolution) != 3:
            print("Locator_resolution argument must be a length-3 tuple -> (cc, ap, lr) voxels")
            exit()

    def _check_pat_fnames(self):
        for pat_fname in self.pat_fnames:
            if ".nii" not in pat_fname:
                print(f"Sorry! Inference is currently only written for nifti (.nii) images...\n found: {pat_fname} in --image_dir")
                exit()
    
    def _check_im(self, min_val):
        if min_val < 0:
            print(f"Expected CT in WM mode (min intensity at 0), instead fname: {pat_fname} min at {im.min()} -> adjusting...")
            return True
        return False

    def run_inference(self):
        # perform inference for all images in directory            
        for pat_idx, pat_fname in enumerate(tqdm(self.pat_fnames)):
            # carries out the full Locator inference and cropping process on a single CT image
            self.nii_im = sitk.ReadImage(join(self.image_dir, pat_fname))
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
            im = resize(im, output_shape=self.Locator_resolution, order=3, preserve_range=True, anti_aliasing=True)
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
            if self.coords_only:
                # store coords in dictionary
                self.coords[pat_fname.replace('.nii','')] = self.rescaled_coords
                continue
            
            # perform cropping around located CoM and save result
            self._apply_crop()
            sitk.WriteImage(self.nii_im, join(self.output_dir, pat_fname))

        if self.coords_only:
            # save all coords
            self._save_coords()

    def inference(self, im):
        # send image to the GPU
        im = torch.tensor(im, dtype=torch.float).to(self.device)
        # New inference using fitted gaussian, come back and better comment this black magic
        # define function to fit (generates a 3D gaussian for a given point [mu_i, mu_j, mu_k] and returns the flattened array)
        def f(t, mu_i, mu_j, mu_k):
            pos = np.array([mu_i, mu_j, mu_k])
            t = t.reshape((3,) + self.Locator_resolution)
            dist_map = np.sqrt(np.sum([np.power((2*(t[0] - pos[0])), 2), np.power((t[1] - pos[1]), 2), np.power((t[2] - pos[2]), 2)], axis=0))
            gaussian = np.array(norm(scale=10).pdf(dist_map), dtype=np.float)
            return gaussian.ravel()
        # run model forward to generate heatmap prediction
        model_output = self.model(im).detach().cpu().numpy()[0]
        # get starting point for curve-fitting (argmax)
        argmax_pred = np.unravel_index(np.argmax(model_output), self.Locator_resolution)
        # do gaussian fitting
        t = np.indices(self.Locator_resolution).astype(np.float)
        p_opt, _ = curve_fit(f, t.ravel(), model_output.ravel(), p0=argmax_pred)
        return p_opt
    
    def _save_coords(self):
        with open(join(self.output_dir, "coords.pkl"), 'wb') as f:
            pickle.dump(self.coords, f)

    def _apply_crop(self):
        # crop the original CT down based upon the Locator CoM coords prediction
        buffers = np.array(self.output_resolution) // 2
        low_crop, hi_crop = self.rescaled_coords - buffers, self.rescaled_coords + buffers
        # cast to int for index slice cropping
        low_crop, hi_crop = np.round(low_crop).astype(int), np.round(hi_crop).astype(int)
        # slice original image to crop - NOTE: nifti image indexing order is lr, ap, cc
        # TODO: add error case where CoM too close to image boundary
        self.nii_im = self.nii_im[low_crop[2]:hi_crop[2], low_crop[1]:hi_crop[1], low_crop[0]:hi_crop[0]]


def setup_argparse():
    parser = ap.ArgumentParser(prog="Main inference program for 3D location-finding network \"Locator\"")
    parser.add_argument("--model_dir", type=str, help="The file path where the model weights are saved to")
    parser.add_argument("--use_attention", default=True, type=lambda x:str2bool(x), help="Doe the model with attention gates?")
    parser.add_argument("--image_dir", type=str, help="The file path of the folder containing the CT images")
    parser.add_argument("--output_dir", type=str, help="The file path where the cropped subvolumes or found CoM values will be saved to")
    parser.add_argument("--subvolumes_or_coords", type=str, choices=['subvolumes', 'coords'], help="Whether the output should be subvolumes or CoM coordinates")
    parser.add_argument("--Locator_resolution", nargs="+", default=[64,256,256], help="Image resolution for Locator, pass in cc, ap, lr order")
    parser.add_argument("--output_resolution", nargs="+", default=[64,128,128], help="The size of the output crop desired (around identified CoM)")
    args = parser.parse_args()
    return args

'''
class headHunter_testing_module:
    def __init__(self, model_dir):
        # setup on class instantiation
        self.model_dir = model_dir
        self.model = self.setup_model()
        # declare class vars
        self.im_ct = None
        self.transforms = None
        self.downscaled_im = None
        self.unscaled_coords = None
        self.scaled_coords = None
        
    def process(self, path_to_ct, gnd_truth_target):
        # carries out the full headHunter inference and cropping process on a single CT image
        nifty_im = sitk.ReadImage(path_to_ct)
        self.spacing = np.array(nifty_im.GetSpacing())
        self.im_ct = np.clip(sitk.GetArrayFromImage(nifty_im), 0, 3024)
        del nifty_im
        self.gnd_truth_target = gnd_truth_target
        self.model = self.setup_model()
        self.transforms = self.transforms_data()
        self.downscaled_im = self.crop_n_downscale()
        self.unscaled_coords = self.inference()
        return self.scale_coords(), self.spacing, self.voxels_away()
    
    def setup_model(self):
        model = headHunter(filter_factor=2)
        for param in model.parameters():
            param.requires_grad = False
        model.to('cuda')
        model.load_best(self.model_dir, logger=None)
        model.eval()
        return model
    
    class transforms_data:
        def __init__(self):
            self.shifts = np.zeros((3)) # only need the low-end shifts
            self.scaling = np.ones((3)) # scale factor of downsizing op
            self.flipped = False        # whether the ct requires CC flip

    def crop_n_downscale(self):
        # Crop and downscale the input ct image - keep track of transformations using the transforms class
        # Initial lr and ap cropping
        downscaled_im = self.im_ct.copy()
        cc_size = self.im_ct.shape[0]
        ap_size = self.im_ct.shape[1]
        lr_size = self.im_ct.shape[2]
        if ap_size > 480 or lr_size > 480:
            ap_over = int((ap_size-480)/2)
            lr_over = int((lr_size-480)/2)
            downscaled_im = downscaled_im[:,ap_over:(ap_size-ap_over-(ap_size%2)),lr_over:(lr_size-lr_over-(lr_size%2))]
            self.transforms.shifts[1] = ap_over
            self.transforms.shifts[2] = lr_over
        
        # Perform thresholding and fill operations
        temp_im = downscaled_im.copy()
        inds = temp_im < (-200+1024)
        temp_im[...] = 1
        temp_im[inds] = 0
        temp_im = binary_fill_holes(temp_im).astype(int)

        # Cut down in the cranio-caudal direction according to the maximimal points of the body id
        filled_inds = np.nonzero(temp_im)
        low_cc_cut = filled_inds[0][0]
        high_cc_cut = filled_inds[0][-1]
        temp_im = temp_im[low_cc_cut:high_cc_cut,:,:]
        downscaled_im = downscaled_im[low_cc_cut:high_cc_cut,:,:]
        self.transforms.shifts[0] = low_cc_cut

        # check if flip required
        if np.mean(temp_im[:5,:,0:120]) > np.mean(temp_im[-5:,:,0:120]):
            downscaled_im = np.flip(downscaled_im, axis=0)
            self.transforms.flipped = True

        # downsampling
        size_before = downscaled_im.shape
        downscaled_im = rescale(downscaled_im, scale=0.5, order=0, multichannel=False, preserve_range=True, anti_aliasing=True)
        downscaled_im = resize(downscaled_im, output_shape=(48,120,120), order=0, preserve_range=True, anti_aliasing=True)
        self.transforms.scaling[0] = 48 / size_before[0]
        self.transforms.scaling[1] = 120 / size_before[1]
        self.transforms.scaling[2] = 120 / size_before[2]

        # preprocess prior to inference
        downscaled_im3 = np.zeros(shape=(1, 3, downscaled_im.shape[0], downscaled_im.shape[1], downscaled_im.shape[2]))
        downscaled_im3[0,0] = windowLevelNormalize(downscaled_im, level=1064, window=350)
        downscaled_im3[0,1] = windowLevelNormalize(downscaled_im, level=1064, window=80)
        downscaled_im3[0,2] = windowLevelNormalize(downscaled_im, level=1624, window=2800)

        return downscaled_im3
    
    def inference(self):
        ''''''
        # Old inference using argmax
        model_output = self.model(torch.tensor(self.downscaled_im, dtype=torch.float).to('cuda')).cpu() ## Might need to add dummy Batch and channels dims
        pred_coords = np.unravel_index(torch.argmax(model_output[0]), model_output.size()[2:])
        return np.array(pred_coords, dtype=float)
        ''''''
        # New inference using fitted gaussian
        model_output = self.model(torch.tensor(self.downscaled_im, dtype=torch.float).to('cuda')).cpu()
        t = np.indices((48,120,120)).astype(float)
        def f(t, mu_i, mu_j, mu_k):
            scale = 10
            pos = np.array([mu_i, mu_j, mu_k])
            t = t.reshape((3,48,120,120))
            dist_map = np.sqrt(np.sum([np.power((2*(t[0] - pos[0])), 2), np.power((t[1] - pos[1]), 2), np.power((t[2] - pos[2]), 2)], axis=0))
            return np.array(norm(scale=10).pdf(dist_map), dtype=float).ravel()
        
        argmax_pred = np.unravel_index(torch.argmax(model_output[0]), model_output.size()[2:])
        popt = curve_fit(f, t.ravel(), model_output[0].numpy().ravel(), p0=argmax_pred)
        return np.array(popt[0], dtype=float)
    
    def scale_coords(self):
        # bump the output coorinates up to match the original input ct
        # also do the same operations to the corresponding target coords for comparison
        scaled_coords = self.unscaled_coords.copy()
        scaled_gndtruth_coords = self.gnd_truth_target.copy()
        # first scale back up
        scaled_coords /= self.transforms.scaling
        scaled_gndtruth_coords /= self.transforms.scaling
        # now pad to account for shifts
        scaled_coords += self.transforms.shifts
        scaled_gndtruth_coords += self.transforms.shifts
        return scaled_coords, scaled_gndtruth_coords

    def voxels_away(self):
        return np.abs(self.unscaled_coords - np.round(self.gnd_truth_target))

'''
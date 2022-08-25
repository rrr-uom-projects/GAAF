# preprocess_Locator_data.py
## generates the training data required for GAAF
from os.path import join
import pickle
import argparse as ap

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from scipy.ndimage import center_of_mass
from tqdm import tqdm

from utils import getFiles, windowLevelNormalize, try_mkdir

class Preprocessor():
    def __init__(self, args, test=False):
        if test:
            self.target_ind = 1
            return
        # setup paths
        self.in_image_dir = args.in_image_dir
        self.in_mask_dir = args.in_mask_dir
        self.output_dir = args.out_image_dir
        self.target_dir = args.CoM_targets_dir
        # try setup output directories
        try_mkdir(dir_name=self.output_dir)
        try_mkdir(dir_name=self.target_dir)
        # find patient fnames
        self.pat_fnames = sorted(getFiles(self.in_image_dir))
        self.mask_fnames = sorted(getFiles(self.in_mask_dir))
        self._check_fnames()
        # setup dictionary for coord targets
        self.CoM_targets = {}
        # setup dictionary for resize performed (for back translation)
        self.resizes_performed = {}
        # set resolution for locator
        self.Locator_image_resolution = tuple([int(res) for res in args.Locator_image_resolution])
        self._check_resolution()
        # set target index
        self.target_ind = args.target_ind

    def run_preprocessing(self):
        for pat_idx, (pat_fname, mask_fname) in enumerate(tqdm(zip(self.pat_fnames, self.mask_fnames), total=len(self.pat_fnames))):
            # load files
            im = sitk.ReadImage(join(self.in_image_dir, pat_fname))
            mask = sitk.ReadImage(join(self.in_mask_dir, mask_fname))
            assert(im.GetSize() == mask.GetSize())
                        
            # Check im direction etc. here and if flip or rot required
            if(np.sign(im.GetDirection()[-1]) == 1):
                im = sitk.Flip(im, [False, False, True])
                mask = sitk.Flip(mask, [False, False, True])

            # change to numpy
            im = sitk.GetArrayFromImage(im)
            if self._check_im(min_val=im.min(), fname=pat_fname):
                im += 1024
            im = np.clip(im, 0, 3024)
            mask = sitk.GetArrayFromImage(mask)
            self._check_mask(mask)
            
            # calculate CoM
            CoM = np.array(center_of_mass(np.array((mask == self.target_ind), dtype=int))) 
            
            # check size of ct here --> resize to desired size if required          
            # resampling
            init_shape = np.array(im.shape)
            im = resize(im, output_shape=self.Locator_image_resolution, order=3, preserve_range=True, anti_aliasing=True)

            # Resize complete
            final_shape = np.array(im.shape)

            # check shapes and calculate new CoM position
            resize_performed = final_shape / init_shape
            CoM *= resize_performed
                        
            # add resize and CoM to respective dictionaries
            self.resizes_performed[pat_fname.replace('.nii','')] = resize_performed
            self.CoM_targets[pat_fname.replace('.nii','')] = CoM

            # finally perform window and level contrast preprocessing on CT -> make this a tuneable feature in future
            # NOTE: Images are forced into WM mode -> i.e. use level = HU + 1024
            # ensure to add extra axis for channels
            im = windowLevelNormalize(im, level=1064, window=1600)[np.newaxis]

            # save CT in numpy format
            np.save(join(self.output_dir, pat_fname.replace('.nii','.npy')), im)
        
        self._save_CoM_targets()
        self._save_resizes()

    def _save_CoM_targets(self):
        with open(join(self.target_dir, "CoM_targets.pkl"), 'wb') as f:
            pickle.dump(self.CoM_targets, f)

    def _load_CoM_targets(self):
        with open(join(self.target_dir, "CoM_targets.pkl"), 'rb') as f:
            self.CoM_targets = pickle.load(f)

    def _save_resizes(self):
        with open(join(self.target_dir, "resizes_performed.pkl"), 'wb') as f:
            pickle.dump(self.resizes_performed, f)
    
    def _check_fnames(self):
        for pat_fname in self.pat_fnames:
            if ".nii" not in pat_fname:
                raise NotImplementedError(f"Sorry! Preprocessing is currently only written for nifti (.nii) images...\n found: {pat_fname} in --in_image_dir")
        # First check all masks in directory are niftis
        for mask_fname in self.mask_fnames:
            if ".nii" not in mask_fname:
                raise NotImplementedError(f"Sorry! Preprocessing is currently only written for nifti (.nii) images...\n found: {mask_fname} in --in_mask_dir")
        # now check that all images and masks are in matching pairs
        for pat_fname in self.pat_fnames:
            if pat_fname not in self.mask_fnames:
                raise ValueError(f"Whoops, it looks like your images and masks aren't in matching pairs!\n found: {pat_fname} in the image directory, but not in the mask directory...")
        for mask_fname in self.mask_fnames:
            if mask_fname not in self.pat_fnames:
                raise ValueError(f"Whoops, it looks like your images and masks aren't in matching pairs!\n found: {pat_fname} in the mask directory, but not in the image directory...")

    def _check_im(self, min_val, fname):
        if min_val < 0:
            print(f"WARNING: Image {fname} has negative values! This is not supported by the current preprocessing pipeline.\nPlease ensure that your images are in Hounsfield Units (HU) + 1024 (WM mode) and that the minimum value is >= 0")
            #print(f"Expected CT in WM mode (min intensity at 0), instead fname: {fname} min at {min_val} -> adjusting...")
            #return True
        return False

    def _check_mask(self, mask):
        if mask.min() != 0:
            raise ValueError("Heyo, we've got a mask issue (min != 0)\nGAAF expects integer masks for the targets with background = 0 and foreground stucture = target_ind ...")
        if (mask == self.target_ind).any() == False:
            raise ValueError("Heyo, we've got a mask issue (target_ind not found in mask)\nGAAF expects integer masks for the targets with background = 0 and foreground stucture = target_ind ...")

    def _check_resolution(self):
        if not isinstance(self.Locator_image_resolution, tuple) or len(self.Locator_image_resolution) != 3:
            raise ValueError("Locator_image_resolution argument must be 3 space-separated integers -> cc ap lr voxels")

def setup_argparse():
    parser = ap.ArgumentParser(prog="Preprocessing program for 3D location-finding network \"Locator\"")
    parser.add_argument("--in_image_dir", type=str, help="The file path containing the raw .nii images")
    parser.add_argument("--in_mask_dir", type=str, help="The file path containing the raw .nii masks")
    parser.add_argument("--out_image_dir", type=str, help="The file path where the resampled images will be saved to")
    parser.add_argument("--CoM_targets_dir", type=str, help="The file path where the coordinate targets and resize info will be saved")
    parser.add_argument("--Locator_image_resolution", nargs="+", default=[64,128,128], help="Image resolution for Locator, pass in cc, ap, lr order")
    parser.add_argument("--target_ind", type=int, default=1, help="The index of the target structure in the mask")
    args = parser.parse_args()
    return args
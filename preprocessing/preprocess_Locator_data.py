# preprocess_Locator_data.py
## generates the training data required for AutoGAF
from os.path import join
import pickle
import argparse as ap

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from scipy.ndimage import center_of_mass
from tqdm import tqdm

from utils.utils import getFiles, windowLevelNormalize, try_mkdir

class Preprocessor():
    def __init__(self, args):
        # setup paths
        self.in_image_dir = args.image_dir
        self.in_struct_dir = args.struct_dir
        self.output_dir = args.output_dir
        self.target_dir = args.target_dir
        # try setup output directories
        try_mkdir(dir_name=self.output_dir)
        try_mkdir(dir_name=self.target_dir)
        # find patient fnames
        self.pat_fnames = sorted(getFiles(self.in_image_dir))
        # setup dictionary for coord targets
        self.CoM_targets = {}
        # setup dictionary for resize performed (for back translation)
        self.resizes_performed = {}
        # set resolution for locator
        self.Locator_resolution = tuple([int(res) for res in args.resolution])
        self._check_resolution()

    def run_preprocessing(self):
        for pat_idx, pat_fname in enumerate(tqdm(self.pat_fnames)):
            # load files
            im = sitk.ReadImage(join(self.in_image_dir, pat_fname))
            mask =sitk.ReadImage(join(self.in_struct_dir, pat_fname))
            assert(im.GetSize() == mask.GetSize())
                        
            # Check im direction etc. here and if flip or rot required
            if(np.sign(im.GetDirection()[-1]) == 1):
                im = sitk.Flip(im, [False, False, True])
                mask = sitk.Flip(mask, [False, False, True])

            # change to numpy
            im = sitk.GetArrayFromImage(im)
            self._check_im(im)
            im = np.clip(im, 0, 3024)
            mask = sitk.GetArrayFromImage(mask)
            self._check_mask(mask)
            
            # calculate CoM
            CoM = np.array(center_of_mass(np.array((mask==2), dtype=int))) 
            
            # check size of ct here --> resize to 256^2 in-plane and x cc             
            # resampling
            init_shape = np.array(im.shape)
            im = resize(im, output_shape=self.Locator_resolution, order=3, preserve_range=True, anti_aliasing=True)

            # Resize complete
            final_shape = np.array(im.shape)

            # check shapes and calculate new CoM position
            resize_performed = final_shape / init_shape
            CoM *= resize_performed
                        
            # add resize and CoM to respective dictionaries
            self.resizes_performed[pat_fname.replace('.nii','')] = resize_performed
            self.CoM_targets[pat_fname.replace('.nii','')] = CoM

            # finally perform window and level contrast preprocessing on CT -> make this a tuneable feature in future
            # NOTE: Images are expected in WM mode -> use level = HU + 1024
            # ensure to add extra axis for channels
            im = windowLevelNormalize(im, level=1064, window=1600)[np.newaxis]
            # im_multi_channel = np.zeros(shape=(num_channels,)+im.shape)
            # im_multi_channel[0] = windowLevelNormalize(im, level=1064, window=1600)
            # im_multi_channel[1] = windowLevelNormalize(im, level= , window= )
            # ...

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
    
    def _check_im(self, im):
        if im.min() < 0:
            print(f"Expected CT in WM mode (min at 0), instead at {im.min()}")
            exit()

    def _check_mask(self, mask):
        try:
            assert(mask.min()==0)
        except AssertionError:
            print("Dodgy mask (min != 0)...")
            exit()
        try:
            assert(mask.max()==2)
        except AssertionError:
            print("Dodgy mask (max != 2)...")
            exit()

    def _check_resolution(self):
        if not isinstance(self.Locator_resolution, tuple) or len(self.Locator_resolution) != 3:
            print("Locator_resolution argument must be a length-3 tuple -> (cc, ap, lr) voxels")
            exit()


def setup_argparse():
    parser = ap.ArgumentParser(prog="Preprocessing program for 3D location-finding network \"Locator\"")
    parser.add_argument("--image_dir", type=str, help="The file path containing the raw .nii images")
    parser.add_argument("--struct_dir", type=str, help="The file path containing the raw .nii masks")
    parser.add_argument("--output_dir", type=str, help="The file path where the resampled images will be saved to")
    parser.add_argument("--target_dir", type=str, help="The file path where the coordinate targets and resize info will be saved")
    parser.add_argument("--resolution", nargs="+", default=[64,256,256], help="Image resolution for Locator, pass in cc, ap, lr order")
    args = parser.parse_args()
    return args
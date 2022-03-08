import pytest
import numpy as np
from ..GAAF.preprocessing_module import Preprocessor

def test_check_resolution():
    # create class instance
    preproc_module = Preprocessor(args=None, test=True)
    # test too many dims specified
    Locator_image_resolution = [64,256,256,1]
    preproc_module.Locator_image_resolution = tuple([int(res) for res in Locator_image_resolution])
    with pytest.raises(ValueError):
        preproc_module._check_resolutions()
    # test too few dims specified
    Locator_image_resolution = [64,256]
    preproc_module.Locator_image_resolution = tuple([int(res) for res in Locator_image_resolution])
    with pytest.raises(ValueError):
        preproc_module._check_resolutions()

def test_check_fnames():
    # create class instance
    preproc_module = Preprocessor(args=None, test=True)
    # test for presence of a non-nifti image
    preproc_module.masks = False
    preproc_module.pat_fnames = ["123.nii","456.nii","789.nrrd","101112.nii"]
    with pytest.raises(NotImplementedError):
        preproc_module._check_fnames()
    # test for presence of a non-nifti mask
    preproc_module.masks = True
    preproc_module.pat_fnames = ["123.nii","456.nii","789.nii","101112.nii"]
    preproc_module.mask_fnames = ["101112.npy","789.nii","456.nii","123.nii"]
    with pytest.raises(NotImplementedError):
        preproc_module._check_fnames()
    # test non-matching set of fnames
    preproc_module.pat_fnames = ["123.nii","456.nii","789.nii","101112.nii"]
    preproc_module.mask_fnames = ["123.nii","789.nii","456.nii"]
    with pytest.raises(ValueError):
        preproc_module._check_fnames()
    # test all good!
    preproc_module.pat_fnames = ["123.nii","456.nii","789.nii"]
    preproc_module.mask_fnames = ["123.nii","789.nii","456.nii"]
    preproc_module._check_fnames()

def test_check_im():
    # create class instance
    preproc_module = Preprocessor(args=None, test=True)
    # check expected cases
    # 1. image in true Houndfield units
    assert(preproc_module._check_im(min_val=-1024))
    # 2. image in WM mode adjusted HU (+1024)
    assert(preproc_module._check_im(min_val=0) == False)

def test_check_mask():
    # create class instance
    preproc_module = Preprocessor(args=None, test=True)
    # check some cases
    # 1. mask is correct
    mask = np.random.uniform(-1, 1, (16,16,16))
    mask[mask<0] = 0
    mask[mask>0] = 1
    preproc_module._check_mask(mask=mask)
    # 2. mask contains floats
    mask = np.random.uniform(-1, 1, (16,16,16))
    with pytest.raises(ValueError):
        preproc_module._check_mask(mask=mask)
    # 3. mask contains multiple structures
    mask = np.random.uniform(-1, 1, (16,16,16))
    mask[mask<0] = 0
    mask[mask>0.66] = 1
    mask[mask>0.33] = 2
    mask[mask>0] = 3
    with pytest.raises(ValueError):
        preproc_module._check_mask(mask=mask)




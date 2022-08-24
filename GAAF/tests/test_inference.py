import pytest
from ..inference_module import Locator_inference_module

def test_check_resolutions():
    # create class instance
    inf_module = Locator_inference_module(args=None, test=True)
    # test too many dims specified
    Locator_image_resolution = [64,256,256,1]
    cropped_image_resolution = [64,256,256]
    inf_module.Locator_image_resolution = tuple([int(res) for res in Locator_image_resolution])
    inf_module.cropped_image_resolution = tuple([int(res) for res in cropped_image_resolution])
    with pytest.raises(ValueError):
        inf_module._check_resolutions()
    # test too few dims specified
    Locator_image_resolution = [64,256,256]
    cropped_image_resolution = [64,256]
    inf_module.Locator_image_resolution = tuple([int(res) for res in Locator_image_resolution])
    inf_module.cropped_image_resolution = tuple([int(res) for res in cropped_image_resolution])
    with pytest.raises(ValueError):
        inf_module._check_resolutions()

def test_check_fnames():
    # create class instance
    inf_module = Locator_inference_module(args=None, test=True)
    # test for presence of a non-nifti image
    inf_module.masks = False
    inf_module.pat_fnames = ["123.nii","456.nii","789.nrrd","101112.nii"]
    with pytest.raises(NotImplementedError):
        inf_module._check_fnames()
    # test for presence of a non-nifti mask
    inf_module.masks = True
    inf_module.pat_fnames = ["123.nii","456.nii","789.nii","101112.nii"]
    inf_module.mask_fnames = ["101112.npy","789.nii","456.nii","123.nii"]
    with pytest.raises(NotImplementedError):
        inf_module._check_fnames()
    # test non-matching set of fnames
    inf_module.pat_fnames = ["123.nii","456.nii","789.nii","101112.nii"]
    inf_module.mask_fnames = ["123.nii","789.nii","456.nii"]
    with pytest.raises(ValueError):
        inf_module._check_fnames()
    # test all good!
    inf_module.pat_fnames = ["123.nii","456.nii","789.nii"]
    inf_module.mask_fnames = ["123.nii","789.nii","456.nii"]
    inf_module._check_fnames()

def test_check_im():
    # create class instance
    inf_module = Locator_inference_module(args=None, test=True)
    # check expected cases
    # 1. image in true Houndfield units
    assert(inf_module._check_im(min_val=-1024, fname="test"))
    # 2. image in WM mode adjusted HU (+1024)
    assert(inf_module._check_im(min_val=0, fname="test") == False)
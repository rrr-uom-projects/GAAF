import pytest
from inference.Locator_module import Locator_inference_module, setup_argparse

def test_check_resolutions():
    # create class instance
    inf_module = Locator_inference_module(args=None, test=True)
    # test too many dims specified
    Locator_image_resolution = [64,256,256,1]
    cropped_image_resolution = [64,256,256]
    inf_module.Locator_resolution = tuple([int(res) for res in Locator_image_resolution])
    inf_module.output_resolution = tuple([int(res) for res in cropped_image_resolution])
    with pytest.raises(ValueError):
        inf_module._check_resolutions()
    # test too few dims specified
    Locator_image_resolution = [64,256,256]
    cropped_image_resolution = [64,256]
    inf_module.Locator_resolution = tuple([int(res) for res in Locator_image_resolution])
    inf_module.output_resolution = tuple([int(res) for res in cropped_image_resolution])
    with pytest.raises(ValueError):
        inf_module._check_resolutions()

def test_check_pat_fnames():
    # create class instance
    inf_module = Locator_inference_module(args=None, test=True)
    # test for presence of a non-nifti image
    inf_module.pat_fnames = ["123.nii","456.nii","789.nrrd","101112.nii"]
    with pytest.raises(NotImplementedError):
        inf_module._check_pat_fnames()

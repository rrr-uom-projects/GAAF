# GAAF: Generalised Automatic Anatomy Finder
## A general framework for 3D location-finding in CT scans
![GAAF_logo](https://user-images.githubusercontent.com/35701423/156757041-887e7937-7e71-4e80-9795-89dc566ae5d7.svg)

The idea of this framework is that it's a fully generalised version of previous specific location-finding models (headHunter, neckNavigator, etc.). With this you can quickly and easily create new location-specific models in just a few lines of code.

The examples are currently set up to generate a brainstem-locating model.

Arxiv paper: https://arxiv.org/abs/2209.06042

## Features

Quickly and easily train location-finding models in just a few lines of code!

## Installation
Install **GAAF** by running:

```
git clone git@github.com:rrr-uom-projects/GAAF.git
```

## What you'll need
1. A training dataset of CT images (nifti format) ~100?
2. Masks of the structures you wish to find the centre-of-mass of for the training images (nifti format)
3. A linux machine with a GPU (ideally one with the 2080ti's or a 3090)

## Instructions
1. Preprocessing
    - The first step to use **GAAF** is to preprocess your training dataset of images and masks into a format which can be used for training. This should be carried out automatically using **GAAF**'s preprocessing module.
    - Check out ```run_preprocessing.sh```. This shell script will run the neccessary preprocessing program. You will need to update the arguments in this script to match the file paths of the directories containing your training images and masks. You will also need to specify your desired output directories. These are the locations where the preprocessed images and centre-of-mass targets for the training process will be saved. Another argument you need to decide on is the image resolution at which to train your custom location-finding model. The maximum size will vary between training rigs. The current default is (64,128,128) which seems to work okay. Finally, remeber to set the target_ind argument to set the index of the structure of interest in your masks.
    - Update the arguments and run ```run_preprocessing.sh```.
2. Training
    - The second stage is to use **GAAF** to train your model!
    - Take a look at ```run_training.sh```. This shell script will run the training. In this script you can specify a load of different choices to customise your model. The most important of these are model_dir (where to save your model), image_dir (where the preprocessed images from stage 1 are saved) and CoM_targets_dir (where the centre-of_mass targets from stage 1 are saved). If training with a smaller GPU (or larger resolution images) you may need to turn down the training and validation batch sizes.
    - Update the arguments and run ```run_training.sh```
3. Inference
    - Now you've successfully trained a model, use **GAAF** to do end-to-end inference!
    - ```run_inference.sh``` is what you'll want for this. Specify the model_dir, and model-specific arguments (as in stage 2). Here you want to set in_image_dir to the directory containing your nifti images for location-finding and out_image_dir to the desired location to save the output cropped images.
    - Most importantly you will also need to specify cropped_image_resolution. This is the size of the subvolume you wish the inference module to crop from your original CT scan.
    - Inference can now be performed with or without masks of structures. If you'd like masks to be cropped simultaneously with your CT images then include the optional arguments in_mask_dir and out_mask_dir. Note - currently **GAAF** requires nifti masks with the same filenames as the CT images, i.e. image-mask pairs like "in_image_dir/0001.nii" and "in_mask_dir/0001.nii"
    - Additionally, if you'd also like the raw coordinates found by the CNN to do the cropping, include the optional out_CoMs_dir argument and provide a location where these can be saved to.
    - Update the arguments and run ```run_inference.sh```

## Some examples of **GAAF** training in action
- A **headHunter** model looking for the centre-of-mass of the brainstem and parotid glands in HN CT scans

![headHunter_sag](https://user-images.githubusercontent.com/35701423/152800962-62db124e-43fb-4e4a-a1e4-f878198cf716.gif)

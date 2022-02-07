raise NotImplementedError()

from headHunter.headHunter_module import headHunter_testing_module
from headHunter.headHunter_utils import k_fold_split_testset_inds
from Training.utils import getFiles
import numpy as np
import os

imagedir = "/data/Jigsaw3D/headHunter_data/uncropped/"
CoM_targets = np.load("/data/Jigsaw3D/headHunter_data/CoM_targets.npy")
outputDir = "/data/Jigsaw3D/headHunter_data/"

# results arrs
results_unrounded = np.zeros((5,22,3,2))
results_rounded = np.zeros((5,22,3,2))
# scrape all im fnames
all_fnames = sorted(getFiles(imagedir))
# iterate over each fold
for fold_num in [5]:#[1,2,3,4,5]:
    # get test im fnames
    test_inds = k_fold_split_testset_inds(len(all_fnames), fold_num=fold_num)
    test_im_fnames = [all_fnames[ind] for ind in test_inds]
    test_targets = [CoM_targets[ind] for ind in test_inds]
    
    # setup inference module
    model_dir = '/data/Jigsaw3D/headHunter_models/new_heatmaps/fold' + str(fold_num) + '/'
    testing_module = headHunter_testing_module(model_dir=model_dir)
    
    # assess performance on test ims in this fold
    for test_im_num, fname in enumerate(test_im_fnames):
        path_to_ct = os.path.join(imagedir, fname)
        gndtruth_target_scaled = test_targets[test_im_num]
        print(f'testing im: {fname}, model: {fold_num}')
        (pred_coords, gndtruth_coords), spacing, voxels_away = testing_module.process(path_to_ct, gndtruth_target_scaled)
        # re-order spacing vector to (cc,ap,lr)
        spacing[[0,1,2]] = spacing[[2,1,0]]
        # test the rounded and non rounded versions
        # unrounded
        diff = spacing * (gndtruth_coords - pred_coords)
        results_unrounded[fold_num-1, test_im_num, :, 0] = diff
        # rounded
        diff = spacing * (gndtruth_coords - np.round(pred_coords).astype(int))
        results_rounded[fold_num-1, test_im_num, :, 0] = diff
        # store the voxels_away for plotting purposes
        results_unrounded[fold_num-1, test_im_num, :, 1] = voxels_away
        results_rounded[fold_num-1, test_im_num, :, 1] = voxels_away
    # attempt garbage collection, properly won't matter
    del testing_module
# save for analysis
np.save(os.path.join(outputDir, 'fittedNormTrainedfold5_unrounded_results.npy'), results_unrounded)
np.save(os.path.join(outputDir, 'fittedNormTrainedfold5_rounded_results.npy'), results_rounded)
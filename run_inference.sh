#!/bin/bash
python3 AutoGAF/inference.py --model_dir "/home/ed/AutoGAF/models/fold2" \
                             --use_attention "False" \
                             --in_image_dir "/home/ed/AutoGAF/data/CTs" \
                             --out_image_dir "/home/ed/AutoGAF/data/test_cropped_cts" \
                             --inference_mode "subvolumes" \
                             --Locator_image_resolution 64 128 128 \
                             --cropped_image_resolution 64 128 128
#!/bin/bash
python3 GAAF/inference.py --model_dir "/home/ed/GAAF/models/fold2" \
                             --use_attention "False" \
                             --in_image_dir "/home/ed/GAAF/data/CTs" \
                             --out_image_dir "/home/ed/GAAF/data/test_cropped_cts" \
                             --inference_mode "subvolumes" \
                             --Locator_image_resolution 64 128 128 \
                             --cropped_image_resolution 64 128 128
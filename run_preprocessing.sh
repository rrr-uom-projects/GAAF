#!/bin/bash
python3 example_preprocessing.py --image_dir "/data/heartHunter/data/CTs"\
                                 --struct_dir "/data/heartHunter/data/Structures"\
                                 --output_dir "/data/heartHunter/data/rescaled_CTs"\
                                 --target_dir "/data/heartHunter/data/"\
                                 --resolution 64 128 128
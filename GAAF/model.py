import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_parts import conv_module, resize_conv, transpose_conv, attention_module

#####################################################################################
#################################  Locator model ####################################
#####################################################################################

class Locator(nn.Module):
    def __init__(self, filter_factor, n_targets, in_channels, p_drop=0.25):
        super(Locator, self).__init__()
        ff = int(filter_factor) # filter factor (easy net scaling)
        # conv layers set 1 - down 1
        self.convs_set_1 = conv_module(in_channels=in_channels, out_channels=32*ff, p_drop=p_drop)
        # conv layers set 2 - down 2
        self.convs_set_2 = conv_module(in_channels=32*ff, out_channels=64*ff, p_drop=p_drop)
        # conv layers set 3 - base
        self.convs_set_3 = conv_module(in_channels=64*ff, out_channels=64*ff, p_drop=p_drop)
        # upsample and layers set 4 - up 1
        self.upsample_1 = resize_conv(in_channels=64*ff, out_channels=64*ff, p_drop=p_drop)
        self.convs_set_4 = conv_module(in_channels=128*ff, out_channels=32*ff, p_drop=p_drop)
        # upsample and layers set 5 - up 2
        self.upsample_2 = resize_conv(in_channels=32*ff, out_channels=32*ff, p_drop=p_drop)
        self.convs_set_5 = conv_module(in_channels=64*ff, out_channels=16*ff, p_drop=p_drop)
        # prediction convolution
        self.pred = nn.Conv3d(in_channels=16*ff, out_channels=n_targets, kernel_size=1)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        # Down block 1
        down1 = self.convs_set_1(x)
        x = F.max_pool3d(down1, (2,2,2))
        # Down block 2
        down2 = self.convs_set_2(x)
        x = F.max_pool3d(down2, (2,2,2))
        # Base block 
        x = self.convs_set_3(x)
        # Upsample and up block 1
        x = self.upsample_1(x)
        x = torch.cat((x, down2), dim=1)
        x = self.convs_set_4(x)
        # Upsample and up block 2
        x = self.upsample_2(x)
        x = torch.cat((x, down1), dim=1)
        x = self.convs_set_5(x)
        # Predict
        return self.pred(x)

    def load_best(self, checkpoint_dir, logger):
        # load previous best weights
        model_dict = self.state_dict()
        state = torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pytorch'))
        best_checkpoint_dict = state['model_state_dict']
        # remove the 'module.' wrapper
        renamed_dict = OrderedDict()
        for key, value in best_checkpoint_dict.items():
            new_key = key.replace('module.','')
            renamed_dict[new_key] = value
        # identify which layers to grab
        renamed_dict = {k: v for k, v in list(renamed_dict.items()) if k in model_dict}
        model_dict.update(renamed_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("Loaded layers from previous best checkpoint:")
            logger.info([k for k, v in list(renamed_dict.items())])


class Attention_Locator(nn.Module):
    def __init__(self, n_targets, in_channels, p_drop=0.25):
        super(Attention_Locator, self).__init__()
        # conv layers set 1 - down 1
        self.convs_set_1 = conv_module(in_channels=in_channels, out_channels=32, p_drop=p_drop)
        # conv layers set 2 - down 2
        self.convs_set_2 = conv_module(in_channels=32, out_channels=64, p_drop=p_drop)
        # conv layers set 3 - base
        self.convs_set_3 = conv_module(in_channels=64, out_channels=64, p_drop=p_drop)
        # upsample, attention and layers set 4 - up 1
        self.upsample_1 = resize_conv(in_channels=64, out_channels=64, p_drop=p_drop)
        self.attention_1 = attention_module(F_g=64, F_l=64, F_int=64)
        self.convs_set_4 = conv_module(in_channels=128, out_channels=32, p_drop=p_drop)
        # upsample, attention and layers set 5 - up 2
        self.upsample_2 = resize_conv(in_channels=32, out_channels=32, p_drop=p_drop)
        self.attention_2 = attention_module(F_g=32, F_l=32, F_int=32)
        self.convs_set_5 = conv_module(in_channels=64, out_channels=16, p_drop=p_drop)

        # prediction convolution
        self.pred = nn.Conv3d(in_channels=16, out_channels=n_targets, kernel_size=1)

    def forward(self,x):
        # Down block 1
        down1 = self.convs_set_1(x)
        x = F.max_pool3d(down1, (2,2,2))
        # Down block 2
        down2 = self.convs_set_2(x)
        x = F.max_pool3d(down2, (2,2,2))
        # Base block 
        x = self.convs_set_3(x)
        # Upsample and up block 1
        x = self.upsample_1(x)
        down2 = self.attention_1(g=down2, x=x)
        x = torch.cat((x, down2), dim=1)
        x = self.convs_set_4(x)
        # Upsample and up block 2
        x = self.upsample_2(x)
        down1 = self.attention_2(g=down1, x=x)
        x = torch.cat((x, down1), dim=1)
        x = self.convs_set_5(x)
        # Predict
        return self.pred(x)

    def load_best(self, checkpoint_dir, logger):
        # load previous best weights
        model_dict = self.state_dict()
        state = torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pytorch'))
        best_checkpoint_dict = state['model_state_dict']
        # remove the 'module.' wrapper
        renamed_dict = OrderedDict()
        for key, value in best_checkpoint_dict.items():
            new_key = key.replace('module.','')
            renamed_dict[new_key] = value
        # identify which layers to grab
        renamed_dict = {k: v for k, v in list(renamed_dict.items()) if k in model_dict}
        model_dict.update(renamed_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("Loaded layers from previous best checkpoint:")
            logger.info([k for k, v in list(renamed_dict.items())])

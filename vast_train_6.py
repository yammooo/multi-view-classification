from train import *

config = {
    "fusion_strategy": "early",
    "fusion_depth": "conv3_block4_out",
    "next_start_layer": "conv4_block1_1_conv",
    "fusion_method": "conv",
}

main(config)
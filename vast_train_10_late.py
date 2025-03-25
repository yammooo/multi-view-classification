from train import *

config = {
    "fusion_strategy": "late",
    "fusion_depth": None,
    "next_start_layer": None,
    "fusion_method": "fc",
}

main(config)
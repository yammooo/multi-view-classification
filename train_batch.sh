python multi-view-classification/train.py '{"backbone_model": "convnextsmall", "fusion_strategy": "score", "fusion_depth": null, "next_start_layer": null, "fusion_method": "sum", "freeze_config": {"freeze_blocks": ["stem", "stage_0"]}}'

rm -rf .cache

python multi-view-classification/train.py '{"backbone_model": "vitb16", "fusion_strategy": "score", "fusion_depth": null, "next_start_layer": null, "fusion_method": "sum", "freeze_config": {"freeze_blocks": []}}'

rm -rf .cache

python multi-view-classification/train.py '{"backbone_model": "resnet152", "fusion_strategy": "score", "fusion_depth": null, "next_start_layer": null, "fusion_method": "sum", "freeze_config": {"freeze_blocks": ["conv1", "conv2", "conv3"]}}'

rm -rf .cache

python multi-view-classification/train.py '{"backbone_model": "efficientnetb0", "fusion_strategy": "score", "fusion_depth": null, "next_start_layer": null, "fusion_method": "sum", "freeze_config": {"freeze_blocks": ["stem", "block1", "block2"]}}'

vastai stop instance 19062359
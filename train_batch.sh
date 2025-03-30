python multi-view-classification/train.py '{"backbone_model": "resnet50", "fusion_strategy": "score", "fusion_depth": null, "next_start_layer": null, "fusion_method": "sum", "freeze_config": {"freeze_blocks": ["conv1","conv2","conv3"]}, "share_weights": "none"}'

rm -rf .cache

python multi-view-classification/train.py '{"backbone_model": "resnet50", "fusion_strategy": "score", "fusion_depth": null, "next_start_layer": null, "fusion_method": "sum", "freeze_config": {"freeze_blocks": ["conv1","conv2","conv3"]}, "share_weights": "first_four"}'

rm -rf .cache

vastai stop instance 19109285
vastai stop instance 19109285
vastai stop instance 19109285
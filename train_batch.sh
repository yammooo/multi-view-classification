python multi-view-classification/train.py '{"backbone_model": "resnet50", "fusion_strategy": "score", "fusion_depth": null, "next_start_layer": null, "fusion_method": "prod", "freeze_config": {"freeze_blocks": ["conv1","conv2","conv3"]}, "share_weights": "none"}'

rm -rf .cache

vastai destroy instance 19126884
vastai destroy instance 19126884
vastai destroy instance 19126884
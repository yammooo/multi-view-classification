from model_helpers import apply_freeze_config

def build_model(config):
    """
    Build model based on configuration.
    Expected keys:
      - fusion_strategy: "early", "late", "score"
      - backbone: e.g., "resnet50", "resnet101", "efficientnet"
      - fusion_method: e.g., "max" or "conv"
      - input_shape: input dimensions
      - num_classes: number of output classes
      - freeze_config: dictionary that defines how to freeze blocks
    """
    fusion_strategy = config.get("fusion_strategy")
    backbone = config.get("backbone")
    fusion_method = config.get("fusion_method")
    input_shape = config.get("input_shape")
    num_classes = config.get("num_classes")
    freeze_config = config.get("freeze_config", None)
    
    if fusion_strategy == "early":
        if backbone == "resnet50":
            from models.early_fusion.resnet50_early import build_5_view_resnet50_early
            model = build_5_view_resnet50_early(input_shape=input_shape,
                                               num_classes=num_classes,
                                               fusion_method=fusion_method)
            if freeze_config:
                apply_freeze_config(model, freeze_config)
            return model
    elif fusion_strategy == "late":
        if backbone == "resnet50":
            from models.late_fusion.resnet50_late import build_5_view_resnet50_late
            model = build_5_view_resnet50_late(input_shape=input_shape,
                                               num_classes=num_classes,
                                               fusion_type=fusion_method)
            if freeze_config:
                apply_freeze_config(model, freeze_config)
            return model
    else:
        raise ValueError("Unknown fusion strategy: " + fusion_strategy)
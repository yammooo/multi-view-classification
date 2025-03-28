from model_helpers import apply_freeze_config

def build_model(config):
    """
    Build multi-view model based on configuration.
    
    Expected keys:
      - fusion_strategy: "early", "late", or "score"
      - backbone_model: e.g., "resnet50", "resnet151", "efficientnetb0", etc.
      - fusion_method: e.g., "max", "conv", "fc", "sum", "prod"
      - input_shape: input dimensions (e.g., (224,224,3))
      - num_classes: number of output classes
      - freeze_config: configuration dict for freezing layers
      - fusion_depth: (for early fusion) the insertion layer name
      - next_start_layer: (for early fusion) the next branch start layer name
    """
    fusion_strategy = config.get("fusion_strategy")
    backbone = config.get("backbone_model")
    fusion_method = config.get("fusion_method")
    input_shape = config.get("input_shape")
    num_classes = config.get("num_classes")
    freeze_config = config.get("freeze_config", None)
    
    if fusion_strategy == "early":
        from models.early_fusion.early_backbone import build_early_backbone
        insertion_layer = config.get("fusion_depth")
        next_start_layer = config.get("next_start_layer")
        model, preprocess_fn = build_early_backbone(
            input_shape=input_shape,
            insertion_layer=insertion_layer,
            next_start_layer=next_start_layer,
            num_classes=num_classes,
            backbone=backbone,
            fusion_method=fusion_method
        )
        if freeze_config:
            apply_freeze_config(model, freeze_config)
        return model, preprocess_fn
    elif fusion_strategy == "late":
        from models.late_fusion.late_backbone import build_late_backbone
        model, preprocess_fn = build_late_backbone(
            input_shape=input_shape,
            num_classes=num_classes,
            backbone=backbone,
            fusion_method=fusion_method
        )
        if freeze_config:
            apply_freeze_config(model, freeze_config)
        return model, preprocess_fn
    elif fusion_strategy == "score":
        from models.score_fusion.score_backbone import build_score_backbone
        model, preprocess_fn = build_score_backbone(
            input_shape=input_shape,
            num_classes=num_classes,
            backbone=backbone,
            fusion_method=fusion_method
        )
        if freeze_config:
            apply_freeze_config(model, freeze_config)
        return model, preprocess_fn
    else:
        raise ValueError("Unknown fusion strategy: " + fusion_strategy)
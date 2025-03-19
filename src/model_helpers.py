import tensorflow as tf
import tensorflow_addons as tfa

def apply_freeze_config(model, freeze_config):
    """
    Freeze blocks in the model based on freeze_config.
    Expected freeze_config example:
      {"freeze_blocks": [1, 2]}  -> Freeze layers whose names contain "1" or "2"
    """
    freeze_blocks = freeze_config.get("freeze_blocks", [])
    for layer in model.layers:
        # If layer name contains any block identifier in freeze_blocks, freeze it.
        if any(block in layer.name for block in freeze_blocks):
            layer.trainable = False
        else:
            # Otherwise set it to trainable.
            layer.trainable = True
    print(f"Applied freeze config: froze blocks {freeze_blocks}")

def get_multi_optimizer(model, base_lr, optimizer_config=None):
    """
    Groups layers based on the '_group' tag and assigns different learning rates.
    
    The optimizer_config should be structured like:
    
      "optimizer_config": {
          "backbone": {"conv1": 0.1, "conv2": 0.2, "conv3": 0.3, "conv4": 0.4, "conv5": 0.5},
          "fusion": 1.0,
          "classifier": 1.0
      }
    
    For the backbone group:
      - If the value is a dict, for each backbone layer, the keys in the dict are matched against
        the layer.name. If a key is found, its multiplier is applied. Otherwise 1.0 is used.
      - If the value is not a dict, that multiplier is used for every backbone layer.
    
    For fusion and classifier (or any non-backbone group) the multiplier is applied directly.
    
    Returns:
      A tfa.optimizers.MultiOptimizer instance.
    """
    # Set default multipliers if not provided.
    if optimizer_config is None:
        optimizer_config = {"backbone": 0.3, "fusion": 1.0, "classifier": 1.0}
    
    # Dictionary to hold groups
    groups = {}
    for layer in model.layers:
        # Get the group from the layer’s _group attribute; default to classifier.
        group = getattr(layer, "_group", "classifier")
        multiplier = 1.0  # Default multiplier
        
        if group == "backbone":
            backbone_config = optimizer_config.get("backbone", 1.0)
            if isinstance(backbone_config, dict):
                # Look for a matching key in the layer name.
                matched = False
                for key, mult in backbone_config.items():
                    if key in layer.name:
                        multiplier = mult
                        matched = True
                        break
                if not matched:
                    multiplier = 1.0  # Use 1.0 if no key found
            else:
                multiplier = backbone_config
        else:
            group_config = optimizer_config.get(group, 1.0)
            # If the value is a dict, use the first value.
            if isinstance(group_config, dict):
                multiplier = list(group_config.values())[0]
            else:
                multiplier = group_config
        
        # Use (group, multiplier) as key to group together layers that share the same setting.
        groups.setdefault((group, multiplier), []).append(layer)
    
    optimizers_and_layers = []
    for (grp, multiplier), layers in groups.items():
        optimizer = tf.keras.optimizers.Adam(base_lr * multiplier)
        optimizers_and_layers.append((optimizer, layers))
    
    multi_optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers=optimizers_and_layers)
    return multi_optimizer
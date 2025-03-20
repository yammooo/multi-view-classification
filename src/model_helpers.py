import tensorflow as tf

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
import tensorflow as tf

def apply_freeze_config(model, freeze_config):
    """
    Recursively freeze blocks in the model based on freeze_config.
    Expected freeze_config example:
      {"freeze_blocks": ["conv1", "conv2"]}  -> Freeze layers whose name contains "conv1" or "conv2"
    """
    freeze_blocks = freeze_config.get("freeze_blocks", [])

    def recursively_freeze(layer):
        # If the layer is a Model, recursively apply to its sub-layers.
        if isinstance(layer, tf.keras.Model):
            for sub_layer in layer.layers:
                recursively_freeze(sub_layer)
        else:
            if any(block in layer.name for block in freeze_blocks):
                layer.trainable = False
            else:
                layer.trainable = True

    for layer in model.layers:
        recursively_freeze(layer)

    print(f"Applied freeze config: froze blocks {freeze_blocks}")
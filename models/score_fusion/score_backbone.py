import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
import keras
import base_model_factory

def create_classifier_branch_model(features_input, num_classes, branch_id, backbone):
    """
    Build a classifier head (as a Model) that maps from features (from the backbone) to predictions.
    Unique layer names are created using branch_id.
    """
    if backbone.lower() == "vitb16":
        pooled = keras.layers.GlobalAveragePooling1D(name=f"gap1d_{branch_id}")(features_input)
    else:
        pooled = keras.layers.GlobalAveragePooling2D(name=f"gap2d_{branch_id}")(features_input)
    
    x = keras.layers.Dense(1024, activation='relu', name=f'fc_1024_{branch_id}')(pooled)
    x = keras.layers.BatchNormalization(name=f"bn_fc_{branch_id}")(x)
    x = keras.layers.Dropout(0.25, name=f"dropout_{branch_id}")(x)
    output = keras.layers.Dense(num_classes, name=f'predictions_{branch_id}')(x)
    return output

def create_branch_model(backbone_model, input_shape, num_classes, branch_id, backbone):
    """
    Create a branch model from the backbone's features to predictions.
    This function builds a mini-model (a classifier head) while the backbone is kept external.
    """
    branch_input = keras.layers.Input(shape=input_shape, name=f"branch_input_{branch_id}")
    # Get features from the backbone.
    features = backbone_model(branch_input)
    # Build classifier head using unique names.
    branch_output = create_classifier_branch_model(features, num_classes, branch_id, backbone)
    return keras.Model(inputs=branch_input, outputs=branch_output, name=f"classifier_branch_{branch_id}")

def build_score_backbone(input_shape=(224, 224, 3), num_classes=5, backbone="resnet50",
                         fusion_method='fc', share_weights="none"):
    """
    share_weights options:
      - "none": each view gets its own backbone and classifier head.
      - "first_four": views 1-4 share the backbone only (each gets a separate classifier head);
                       view 5 gets its own backbone and classifier head.
      - "all": all 5 views share the same backbone (each gets a separate classifier head).
    """
    input_views = []
    branch_outputs = []
    
    # In the cases where we share part of the network, create a shared backbone instance.
    if share_weights in ["all", "first_four"]:
        shared_backbone, preprocess_fn = base_model_factory.base_model(backbone, input_shape, include_top=False)
        shared_backbone.trainable = True
        # Optionally tag backbone layers.
        for layer in shared_backbone.layers:
            layer._group = "backbone"
    
    if share_weights == "all":
        # All views use the same backbone.
        for i in range(5):
            inp = keras.layers.Input(shape=input_shape, name=f'input_view_{i+1}')
            input_views.append(inp)
            # Instead of applying the shared backbone and then classifier head directly,
            # we build a branch model that wraps the classifier head only.
            branch_model = create_branch_model(shared_backbone, input_shape, num_classes, branch_id=f"v{i+1}", backbone=backbone)
            branch_out = branch_model(inp)
            print(f"View {i+1} (shared backbone) branch output shape:", branch_out.shape)
            branch_outputs.append(branch_out)
    elif share_weights == "first_four":
        # First four views share the backbone.
        for i in range(4):
            inp = keras.layers.Input(shape=input_shape, name=f'input_view_{i+1}')
            input_views.append(inp)
            branch_model = create_branch_model(shared_backbone, input_shape, num_classes, branch_id=f"v{i+1}", backbone=backbone)
            branch_out = branch_model(inp)
            print(f"View {i+1} (shared backbone) branch output shape:", branch_out.shape)
            branch_outputs.append(branch_out)
        # Fifth view gets a separate backbone.
        inp = keras.layers.Input(shape=input_shape, name='input_view_5')
        input_views.append(inp)
        separate_backbone, _ = base_model_factory.base_model(backbone, input_shape, include_top=False)
        separate_backbone.trainable = True
        for layer in separate_backbone.layers:
            layer._group = "backbone"
        branch_model = create_branch_model(separate_backbone, input_shape, num_classes, branch_id="v5", backbone=backbone)
        branch_out = branch_model(inp)
        print("View 5 (separate backbone) branch output shape:", branch_out.shape)
        branch_outputs.append(branch_out)
    else:  # "none": no sharing at all; each view gets its own backbone and classifier head.
        preprocess_fn = None  # Get preprocess_fn from first branch.
        for i in range(5):
            inp = keras.layers.Input(shape=input_shape, name=f'input_view_{i+1}')
            input_views.append(inp)
            branch_backbone, branch_preprocess_fn = base_model_factory.base_model(backbone, input_shape, include_top=False)
            branch_backbone.trainable = True
            for layer in branch_backbone.layers:
                layer._group = "backbone"
            # Build a branch model using this backbone.
            branch_model = create_branch_model(branch_backbone, input_shape, num_classes, branch_id=f"v{i+1}", backbone=backbone)
            branch_out = branch_model(inp)
            print(f"View {i+1} (independent branch) output shape:", branch_out.shape)
            branch_outputs.append(branch_out)
            if i == 0:
                preprocess_fn = branch_preprocess_fn

    # Fuse the outputs from each branch.
    if fusion_method == 'sum':
        fused = keras.layers.Add()(branch_outputs)
    elif fusion_method == 'prod':
        fused = keras.layers.Multiply()(branch_outputs)
    elif fusion_method == 'max':
        fused = keras.layers.Maximum()(branch_outputs)
    else:
        raise ValueError("Unknown fusion_type. Use 'sum', 'prod' or 'max'.")
    
    final_output = keras.layers.Activation('softmax', name='final_softmax')(fused)
    multi_view_model = keras.Model(inputs=input_views, outputs=final_output)
    return multi_view_model, preprocess_fn

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    
    from model_helpers import apply_freeze_config

    # Test with weight sharing among first four views.
    model, preprocessing_fn = build_score_backbone(
        num_classes=5,
        backbone="resnet50",
        fusion_method="sum",
        share_weights="none"  # Options: "none", "first_four", "all"
    )

    model.summary()

    # def print_trainable_status(model_instance, prefix=""):
    #     for layer in model_instance.layers:
    #         if isinstance(layer, tf.keras.Model):
    #             print_trainable_status(layer, prefix=prefix + "  ")
    #         else:
    #             print(f"{prefix}{layer.name}: trainable={layer.trainable}")

    # print("\nBefore applying freeze config:")
    # print_trainable_status(model)

    # from model_helpers import apply_freeze_config
    # freeze_config = {"freeze_blocks": []}
    # apply_freeze_config(model, freeze_config)

    # print("\nAfter applying freeze config:")
    # print_trainable_status(model)
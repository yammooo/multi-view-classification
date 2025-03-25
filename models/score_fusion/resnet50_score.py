import tensorflow as tf
import keras

def create_view_branch(input_shape, num_classes):
    # Load ResNet50 without the classifier part.
    base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True

    for layer in base_model.layers:
        layer._group = "backbone"

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)

    x = keras.layers.Dense(1024, activation='relu', name='fc_1024')(x)

    bn_layer = keras.layers.BatchNormalization(name="bn_fc")
    bn_layer._group = "classifier"
    x = bn_layer(x)

    x = keras.layers.Dropout(0.25)(x)
    x._group = "classifier"

    # Create the branch model.
    output = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    output._group = "classifier"

    branch_model = keras.Model(inputs=base_model.input, outputs=output)
    return branch_model

def build_5_view_resnet50_score(input_shape=(224, 224, 3), num_classes=5, fusion_method='fc'):
    input_views = []
    branch_outputs = []
    
    # Process each of the 5 views
    for i in range(5):
        inp = keras.layers.Input(shape=input_shape, name=f'input_view_{i+1}')
        input_views.append(inp)
        branch = create_view_branch(input_shape, num_classes)
        branch_out = branch(inp)
        print(f"Branch {i+1} output shape:", branch_out.shape)
        branch_outputs.append(branch_out)
    
    # Fuse the branch outputs according to the selected fusion type.
    if fusion_method == 'sum':
        fused = keras.layers.Add()(branch_outputs)
    elif fusion_method == 'prod':
        fused = keras.layers.Multiply()(branch_outputs)
    elif fusion_method == 'max':
        fused = keras.layers.Maximum()(branch_outputs)
    else:
        raise ValueError("Unknown fusion_type. Use 'sum', 'prod' or 'max'.")
    
    # Apply softmax so that the output sums to 1.
    output = keras.layers.Activation('softmax', name='final_softmax')(fused)

    # Create the multi-view model with 5 inputs.
    multi_view_model = keras.Model(inputs=input_views, outputs=output)
    return multi_view_model

if __name__ == "__main__":

    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

    from model_helpers import apply_freeze_config

    # Build the model
    model = build_5_view_resnet50_score(
        num_classes=5,
        fusion_method="prod"
    )
    model.summary()
    
    # Function to print each layer's trainable status recursively.
    def print_trainable_status(model_instance, prefix=""):
        for layer in model_instance.layers:
            if isinstance(layer, tf.keras.Model):
                print_trainable_status(layer, prefix=prefix + "  ")
            else:
                print(f"{prefix}{layer.name}: trainable={layer.trainable}")
    
    print("\nBefore applying freeze config:")
    print_trainable_status(model)
    
    # Import and apply a freeze configuration from model_helpers.
    # For example, freezing any layer whose name contains "conv1", "conv2", etc.
    from model_helpers import apply_freeze_config
    freeze_config = {"freeze_blocks": ["conv1", "conv2", "conv3", "conv4", "conv5"]}
    apply_freeze_config(model, freeze_config)
    
    print("\nAfter applying freeze config:")
    print_trainable_status(model)
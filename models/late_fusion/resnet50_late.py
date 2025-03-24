import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class StackReduceLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(StackReduceLayer, self).__init__(**kwargs)
    def call(self, inputs):
        stacked = tf.stack(inputs, axis=1)
        return tf.reduce_max(stacked, axis=1)
    def get_config(self):
        base_config = super(StackReduceLayer, self).get_config()
        return base_config

def create_view_branch(input_shape):
    # Load ResNet50 without the classifier part
    base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True

    for layer in base_model.layers:
        layer._group = "backbone"

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(base_model.output)

    x = tf.keras.layers.Dense(1024, activation='relu', name='fc_1024')(x)
    x._group = "classifier"

    bn_layer = keras.layers.BatchNormalization(name="bn_fc")
    bn_layer._group = "classifier"
    x = bn_layer(x)

    x = keras.layers.Dropout(0.5)(x)
    x._group = "classifier"

    branch_model = keras.Model(inputs=base_model.input, outputs=x)
    return branch_model

def build_5_view_resnet50_late(input_shape=(224, 224, 3),
                               num_classes=5,
                               fusion_method='fc'):
    
    input_views = []
    branch_outputs = []
    
    # Process each of the 5 views
    for i in range(5):
        inp = keras.layers.Input(shape=input_shape, name=f'input_view_{i+1}')
        input_views.append(inp)
        branch = create_view_branch(input_shape)
        branch_out = branch(inp)
        print(f"Branch {i+1} output shape:", branch_out.shape)
        branch_outputs.append(branch_out)
    
    if fusion_method == 'max':

        x = StackReduceLayer(name="stack_reduce_layer")
        x._group = "fusion"
        fused = x(branch_outputs)
        
        print("Shape after max fusion:", fused.shape)
        x = fused
    elif fusion_method == 'fc':
        # Late fusion (fc): Concatenate feature vectors and fuse using a FC layer.
        fused = keras.layers.Concatenate(axis=-1)(branch_outputs)
        print("Shape after concatenation:", fused.shape)
        x = keras.layers.Dense(1024, activation='relu')(fused)

        bn_layer = keras.layers.BatchNormalization(name="bn_fc")
        bn_layer._group = "classifier"
        x = bn_layer(x)

        x = keras.layers.Dropout(0.5)(x)
        x._group = "classifier"
    else:
        raise ValueError("Unknown fusion_type. Use 'fc' or 'max'.")
    
    # Final classification layer
    output = keras.layers.Dense(num_classes, activation='softmax', name="prediction")(x)
    output._group = "classifier"

    multi_view_model = keras.Model(inputs=input_views, outputs=output)
    return multi_view_model

if __name__ == "__main__":

    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

    from model_helpers import apply_freeze_config

    # Build the model
    model = build_5_view_resnet50_late(
        num_classes=5,
        fusion_method="fc"
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
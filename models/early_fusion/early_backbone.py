import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
import keras
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, Layer, BatchNormalization

import base_model_factory

@keras.saving.register_keras_serializable()
class StackReduceLayer(Layer):
    def __init__(self, **kwargs):
        super(StackReduceLayer, self).__init__(**kwargs)
    def call(self, inputs):
        stacked = tf.stack(inputs, axis=1)
        return tf.reduce_max(stacked, axis=1)
    def get_config(self):
        base_config = super(StackReduceLayer, self).get_config()
        return base_config

def split_backbone(backbone, insertion_layer_name, next_start_layer_name, input_shape=(512, 512, 3)):
    full_model = base_model_factory.base_model(backbone, input_shape, include_top=False)

    full_model.trainable = True

    part1 = Model(
        inputs=full_model.input,
        outputs=full_model.get_layer(insertion_layer_name).output
    )
    
    for layer in part1.layers:
        layer._group = "backbone"

    part2 = None

    if next_start_layer_name is not None:
        part2 = Model(
            inputs=full_model.get_layer(next_start_layer_name).input,
            outputs=full_model.output
        )

        for layer in part2.layers:
            layer._group = "backbone"
    
    return part1, part2

def build_early_backbone(input_shape=(224, 224, 3),
                         insertion_layer="conv2_block3_out",
                         next_start_layer="conv3_block1_1_conv",
                         num_classes=5,
                         backbone="resnet50",
                         fusion_method="max"):
    part1, part2 = split_backbone(backbone, insertion_layer, next_start_layer, input_shape=input_shape)
    
    input_views = []
    branch_outputs = []
    
    for i in range(5):
        inp = Input(shape=input_shape, name=f'input_view_{i+1}')
        input_views.append(inp)
        branch_out = part1(inp)
        branch_outputs.append(branch_out)
        print(f"Branch {i+1} output shape: {branch_out.shape}")
    
    if fusion_method == "max":
        fusion_layer = StackReduceLayer(name="stack_reduce_layer")
        fusion_layer._group = "fusion"
        fused = fusion_layer(branch_outputs)
    elif fusion_method == "conv":
        fused_concat = Concatenate(axis=-1, name="fused_concat")(branch_outputs)
        filters = branch_outputs[0].shape[-1]
        conv_layer = Conv2D(filters=filters,
                            kernel_size=1,
                            activation='relu',
                            name='fused_adapter')
        conv_layer._group = "fusion"
        fused = conv_layer(fused_concat)
    else:
        raise ValueError("fusion_method must be either 'max' or 'conv'")
    
    print("Shape after fusion:", fused.shape)

    if part2:
        x = part2(fused)
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(fused)
    
    fc_layer = tf.keras.layers.Dense(
        1024,
        activation='relu',
        name='fc_1024',
    )
    
    fc_layer._group = "classifier"
    x = fc_layer(x)
    
    bn_layer = BatchNormalization(name="bn_fc")
    bn_layer._group = "classifier"
    x = bn_layer(x)
    
    dropout_layer = keras.layers.Dropout(0.2, name="dropout")
    dropout_layer._group = "classifier"
    x = dropout_layer(x)
    
    pred_layer = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        name="predictions"
    )
    pred_layer._group = "classifier"
    output = pred_layer(x)
    
    multi_view_model = Model(inputs=input_views, outputs=output)
    return multi_view_model

if __name__ == "__main__":

    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

    from model_helpers import apply_freeze_config

    # Build the model
    model = build_early_backbone(
        insertion_layer="block2b_add",
        next_start_layer="block3a_expand_conv",
        num_classes=5,
        backbone="efficientnetb0",
        fusion_method="conv"
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
    freeze_config = {"freeze_blocks": ["conv1", "conv2", "conv3"]}
    apply_freeze_config(model, freeze_config)
    
    print("\nAfter applying freeze config:")
    print_trainable_status(model)
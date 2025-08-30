import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, Layer, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
import keras

class StackReduceLayer(Layer):
    def __init__(self, **kwargs):
        super(StackReduceLayer, self).__init__(**kwargs)
    def call(self, inputs):
        # Stack object: expects a list of tensors.
        stacked = tf.stack(inputs, axis=1)
        # Reduce along the new axis (e.g., take maximum).
        return tf.reduce_max(stacked, axis=1)
    def get_config(self):
        base_config = super(StackReduceLayer, self).get_config()
        return base_config

def split_resnet50(insertion_layer_name, next_start_layer_name, input_shape=(512, 512, 3)):
    # Load the full model with pretrained weights.
    full_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    # Freeze the first three blocks.
    full_model.trainable = True
    for layer in full_model.layers:
        if layer.name.startswith("conv1_") or layer.name.startswith("conv2_") or layer.name.startswith("conv3_"):
            layer.trainable = False
    
    part1 = Model(
        inputs=full_model.input,
        outputs=full_model.get_layer(insertion_layer_name).output
    )
    part2 = Model(
        inputs=full_model.get_layer(next_start_layer_name).input,
        outputs=full_model.output
    )
    return part1, part2

def build_5_view_resnet50_early(input_shape=(224, 224, 3),
                                insertion_layer="conv2_block3_out",
                                next_start_layer="conv3_block1_1_conv",
                                num_classes=5,
                                fusion_type="max"):
    part1, part2 = split_resnet50(insertion_layer, next_start_layer, input_shape=input_shape)
    
    input_views = []
    branch_outputs = []
    
    for i in range(5):
        inp = Input(shape=input_shape, name=f'input_view_{i+1}')
        input_views.append(inp)
        branch_out = part1(inp)
        branch_outputs.append(branch_out)
        print(f"Branch {i+1} output shape: {branch_out.shape}")
    
    if fusion_type == "max":
        fused = StackReduceLayer(name="stack_reduce_layer")(branch_outputs)
    elif fusion_type == "conv":
        fused_concat = Concatenate(axis=-1)(branch_outputs)
        filters = branch_outputs[0].shape[-1]
        fused = Conv2D(filters=filters,
                       kernel_size=1,
                       activation='relu',
                       name='fused_adapter',
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4)
                       )(fused_concat)
    else:
        raise ValueError("fusion_type must be either 'max' or 'conv'")
    
    print("Shape after fusion:", fused.shape)
    
    x = part2(fused)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', name='fc_1024',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    
    multi_view_model = Model(inputs=input_views, outputs=output)
    return multi_view_model

if __name__ == "__main__":
    model = build_5_view_resnet50_early(
        insertion_layer="conv2_block3_out",
        next_start_layer="conv3_block1_1_conv",
        num_classes=5,
        fusion_type="max")
    model.summary()
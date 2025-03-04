from tensorflow import keras
import tensorflow as tf
from keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D, Dropout
from keras.applications import ResNet50

def create_view_branch(input_shape, weights='imagenet'):
    # Load ResNet50 without the classifier part
    base_model = ResNet50(weights=weights, include_top=False, input_shape=input_shape)
    base_model.trainable = False
    # Use global average pooling to obtain a feature vector of dimension 2048
    x = GlobalAveragePooling2D()(base_model.output)
    branch_model = keras.Model(inputs=base_model.input, outputs=x)
    return branch_model

def build_multi_view_model(input_shape=(512, 512, 3), num_classes=10, fusion_type='fc'):
    input_views = []
    branch_outputs = []
    
    # Process each of the 5 views
    for i in range(5):
        inp = Input(shape=input_shape, name=f'input_view_{i+1}')
        input_views.append(inp)
        branch = create_view_branch(input_shape)
        branch_out = branch(inp)
        print(f"Branch {i+1} output shape:", branch_out.shape)  # Expected: (None, 2048)
        branch_outputs.append(branch_out)
    
    if fusion_type == 'fc':
        # Late fusion (fc): Concatenate feature vectors and fuse using a FC layer.
        # Each branch provides a 2048-dimensional vector → final concatenated vector is (None, 2048*5)
        fused = Concatenate(axis=-1)(branch_outputs)
        print("Shape after concatenation:", fused.shape)
        x = Dense(1024, activation='relu')(fused)
        x = Dropout(0.5)(x)
    elif fusion_type == 'max':
        # Late fusion (max): Stack feature vectors and perform element-wise maximum pooling.
        stacked = tf.stack(branch_outputs, axis=1)  # shape: (None, 5, 2048)
        fused = tf.reduce_max(stacked, axis=1)        # shape: (None, 2048)
        print("Shape after max fusion:", fused.shape)
        x = fused
    else:
        raise ValueError("Unknown fusion_type. Use 'fc' or 'max'.")
    
    # Final classification layer
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create the multi-view model with 5 inputs
    multi_view_model = keras.Model(inputs=input_views, outputs=output)
    return multi_view_model

# Build and display model using the late fusion fc approach
multi_view_model = build_multi_view_model(fusion_type='fc')
multi_view_model.summary()
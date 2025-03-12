import tensorflow as tf
import keras

def create_view_branch(input_shape):
    # Load ResNet50 without the classifier part
    base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(1024, activation='relu', name='fc_1024')(x)
    x = keras.layers.Dropout(0.5)(x)
    branch_model = keras.Model(inputs=base_model.input, outputs=x)
    return branch_model

def build_5_view_resnet50_late(input_shape=(512, 512, 3), num_classes=5, fusion_type='fc'):
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
    
    if fusion_type == 'fc':
        # Late fusion (fc): Concatenate feature vectors and fuse using a FC layer.
        fused = keras.layers.Concatenate(axis=-1)(branch_outputs)
        print("Shape after concatenation:", fused.shape)
        x = keras.layers.Dense(1024, activation='relu')(fused)
        x = keras.layers.Dropout(0.5)(x)
    elif fusion_type == 'max':
        # Late fusion (max): Stack feature vectors and perform element-wise maximum pooling.
        stacked = tf.stack(branch_outputs, axis=1)
        fused = tf.reduce_max(stacked, axis=1)
        print("Shape after max fusion:", fused.shape)
        x = fused
    else:
        raise ValueError("Unknown fusion_type. Use 'fc' or 'max'.")
    
    # Final classification layer
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    multi_view_model = keras.Model(inputs=input_views, outputs=output)
    return multi_view_model

# Build and display model using
multi_view_model = build_5_view_resnet50_late(fusion_type='max')
multi_view_model.summary()
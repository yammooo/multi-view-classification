import tensorflow as tf
from tensorflow import keras

def create_view_branch(input_shape, num_classes):
    # Load ResNet50 without the classifier part.
    base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True
    # Use global average pooling to obtain a feature vector.
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    # Add a fully connected layer with 1024 units.
    x = tf.keras.layers.Dense(1024, activation='relu', name='fc_1024')(x)
    x = keras.layers.Dropout(0.5)(x)
    # Create the branch model.
    output = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    branch_model = keras.Model(inputs=base_model.input, outputs=output)
    return branch_model

def build_5_view_resnet50_score(input_shape=(512, 512, 3), num_classes=5, fusion_type='fc'):
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
    if fusion_type == 'sum':
        output = keras.layers.Add()(branch_outputs)
    elif fusion_type == 'prod':
        output = keras.layers.Multiply()(branch_outputs)
    elif fusion_type == 'max':
        output = keras.layers.Maximum()(branch_outputs)
    else:
        raise ValueError("Unknown fusion_type. Use 'sum', 'prod' or 'max'.")
    
    # Create the multi-view model with 5 inputs.
    multi_view_model = keras.Model(inputs=input_views, outputs=output)
    return multi_view_model

# Build and display model
multi_view_model = build_5_view_resnet50_score(fusion_type='sum')
multi_view_model.summary()
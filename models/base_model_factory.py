import keras
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def base_model(backbone, input_shape, include_top=False):
    # Choose the backbone model based on the 'backbone' parameter.
    backbone = backbone.lower()
    if backbone == "resnet50":
        base_model = keras.applications.ResNet50(weights='imagenet', include_top=include_top, input_shape=input_shape)
        preprocess_fn = keras.applications.resnet.preprocess_input
    elif backbone == "resnet152":
        base_model = keras.applications.ResNet152(weights='imagenet', include_top=include_top, input_shape=input_shape)
        preprocess_fn = keras.applications.resnet.preprocess_input
    elif backbone == "efficientnetb0":
        base_model = keras.applications.EfficientNetB0(weights='imagenet', include_top=include_top, input_shape=input_shape)
        preprocess_fn = keras.applications.efficientnet.preprocess_input
    elif backbone == "efficientnetb7":
        base_model = keras.applications.EfficientNetB7(weights='imagenet', include_top=include_top, input_shape=input_shape)
        preprocess_fn = keras.applications.efficientnet.preprocess_input
    elif backbone == "convnextbase":
        base_model = keras.applications.ConvNeXtBase(weights="imagenet", include_top=include_top, input_shape=input_shape)
        preprocess_fn = None
    elif backbone == "convnextsmall":
        base_model = keras.applications.ConvNeXtSmall(weights="imagenet", include_top=include_top, input_shape=input_shape)
        preprocess_fn = None
    elif backbone == "convnexttiny":
        base_model = keras.applications.ConvNeXtTiny(weights="imagenet", include_top=include_top, input_shape=input_shape)
        preprocess_fn = None
    elif backbone == "swintiny":
        from tfswin import SwinTransformerTiny224
        # Note: tfswin models support variable input shapes through built-in preprocessing.
        # For custom pipelines, the model expects uint8 inputs.
        base_model = SwinTransformerTiny224(include_top=include_top)
        # Define a simple preprocessing function: ensure pixel values are clipped to [0, 255] and cast to uint8.
        preprocess_fn = lambda x: np.clip(x, 0, 255).astype('uint8')
    else:
        raise ValueError("Unsupported backbone")
    
    return base_model, preprocess_fn
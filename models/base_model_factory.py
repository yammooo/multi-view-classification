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
    elif backbone == "resnet101":
        base_model = keras.applications.ResNet101(weights='imagenet', include_top=include_top, input_shape=input_shape)
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
    elif backbone == "vitb16":

        import keras_hub
        # Instead of returning a classifier, load only the backbone feature extractor.
        vit_backbone = keras_hub.models.Backbone.from_preset("vit_base_patch16_224_imagenet")
        # Create an input layer matching your input_shape.
        input_layer = keras.layers.Input(shape=input_shape)
        # Pass the input through the ViT backbone.
        features = vit_backbone(input_layer)
        # In some versions the backbone may return a dict; extract the default output if so.
        if isinstance(features, dict):
            features = features.get("default", list(features.values())[0])
        # Create a functional model that outputs the features.
        base_model = keras.Model(inputs=input_layer, outputs=features)

        preprocess_fn = keras_hub.models.ViTImageClassifierPreprocessor.from_preset(
            "vit_base_patch16_224_imagenet"
        )
    elif backbone == "swintiny":
        from tfswin import SwinTransformerTiny224
        # Note: tfswin models support variable input shapes through built-in preprocessing.
        # For custom pipelines, the model expects uint8 inputs.
        base_model = SwinTransformerTiny224(include_top=include_top)
        
        # Simple preprocessing function: ensure pixel values are clipped to [0, 255] and cast to uint8.
        preprocess_fn = lambda x: np.clip(x, 0, 255).astype('uint8')
    else:
        raise ValueError("Unsupported backbone")
    
    return base_model, preprocess_fn
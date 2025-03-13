import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from data_generator import SimpleMultiViewDataGenerator
from evaluation import evaluate_and_log_model
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from models.early_fusion.resnet50_early import build_5_view_resnet50_early

np.random.seed(42)
tf.random.set_seed(42)

def main():

    # ------------------- Configuration -------------------

    wandb.init(
        project="5-view-classification",
        config={
            "dataset_artifact": "synt_5_obj_dataset:v0",
            "input_shape": (224, 224, 3),
            "batch_size": 8,
            "epochs": 1,
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "backbone_model": "resnet50",
            "loss": "categorical_crossentropy",
            "fusion_type": "early",
            "fusion_depth": "conv2_block3_out",
            "fusion_method": "max",
        }
    )
    config = wandb.config
    
    data_dir = r"/home/yammo/C:/Users/gianm/Development/blender-dataset-gen/data/output"
    input_shape = config.input_shape
    batch_size = config.batch_size


    # ------------------- Data Preparation -------------------
    
    print("Initializing data generator...")
    data_gen = SimpleMultiViewDataGenerator(
        data_dir=data_dir,
        input_shape=input_shape,
        batch_size=batch_size
    )
    
    train_ds = data_gen.get_train_dataset()
    test_ds = data_gen.get_test_dataset()
    class_names = data_gen.get_class_names()
    num_classes = data_gen.get_num_classes()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"results/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Visualizing training data...")
    aug_fig = data_gen.visualize_batch()
    aug_fig.savefig(os.path.join(output_dir, "augmented_samples.png"))
    plt.close(aug_fig)


    # ------------------- Building Model -------------------

    print("Building and compiling model...")
    model = build_5_view_resnet50_early(
        input_shape=input_shape,
        num_classes=num_classes,
        fusion_type=config.fusion_method,
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss=config.loss,
        metrics=['accuracy']
    )

    steps_per_epoch = len(data_gen.train_samples) // batch_size
    if len(data_gen.train_samples) % batch_size != 0:
        steps_per_epoch += 1
    
    validation_steps = len(data_gen.test_samples) // batch_size
    if len(data_gen.test_samples) % batch_size != 0:
        validation_steps += 1

    # ------------------- Callbacks -------------------

    callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, 'model_best.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            WandbMetricsLogger(log_freq=5),
            WandbModelCheckpoint(os.path.join(output_dir, "wandb_model_best.keras"))
        ]

    # ------------------- Training and Evaluation on Base Dataset -------------------
    
    print("Training model...")

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        steps_per_epoch=10,
        validation_steps=10
    )

    model.save(os.path.join(output_dir, 'model_final.keras'))
        
    print("Evaluating model on test dataset...")
    evaluate_and_log_model(model, output_dir, config.dataset_artifact, test_ds, None, config, class_names, validation_steps)

    # ------------------- Evaluation on Real Dataset -------------------

    real_data_dir = r"/home/yammo/C:/Users/gianm/Development/multi-view-classification/dataset/test"
    
    evaluate_and_log_model(model, output_dir, "real_obj_dataset:v0", None, real_data_dir, config, None, None)

    # ------------------- Finish -------------------
    
    print(f"All results saved to {output_dir}")
    wandb.finish()
    
if __name__ == "__main__":
    main()

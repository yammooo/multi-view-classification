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

from model_factory import build_model
from model_helpers import get_multi_optimizer

np.random.seed(42)
tf.random.set_seed(42)

def main():

    # ------------------- Configuration -------------------

    wandb.init(
        project="5-view-classification",
        config={
            "dataset_artifact": "synt+real_75+5_dataset:v0",
            "input_shape": (224, 224, 3),
            "batch_size": 16,
            "epochs": 10,
            "optimizer": "adam",
            "backbone_model": "resnet50",
            "loss": "categorical_crossentropy",

            "fusion_strategy": "early",
            "fusion_depth": "conv2_block3_out",
            "next_start_layer": "conv3_block1_1_conv",
            "fusion_method": "max",
            "freeze_config": {"freeze_blocks": []},

            "learning_rate": 1e-4,
            "differential_lr": True,
            "optimizer_config": {
                "backbone": {"conv1": 0.1, "conv2": 0.2, "conv3": 0.3, "conv4": 0.4, "conv5": 0.5},
                "fusion": 1.0,
                "classifier": 1.0
            }
        }
    )
    config = wandb.config
    
    base_dir = os.path.dirname(os.path.join(__file__, ".."))
    ds_dir = os.path.join(base_dir, "synt+real_75+5_dataset_v0")
    input_shape = config.input_shape
    batch_size = config.batch_size


    # ------------------- Data Preparation -------------------
    
    print("Initializing data generator...")
    data_gen = SimpleMultiViewDataGenerator(
        data_dir=ds_dir,
        input_shape=input_shape,
        batch_size=batch_size
    )
    
    train_ds = data_gen.get_train_dataset()
    test_ds = data_gen.get_test_dataset()
    class_names = data_gen.get_class_names()
    num_classes = data_gen.get_num_classes()

    config.num_classes = num_classes
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"results/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    training_batch_fig = data_gen.visualize_batch()
    wandb.log({"training_batch_fig": wandb.Image(training_batch_fig)})

    # ------------------- Building Model -------------------

    print("Building model via factory...")
    model = build_model(config)
    
    # Optimizer
    if config.differential_lr:
        optimizer = get_multi_optimizer(model, config.learning_rate, config.optimizer_config)
    else:
        optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
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
            WandbMetricsLogger(log_freq=5),
            WandbModelCheckpoint(
                filepath= os.path.join(output_dir, "model_best.keras"),
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            # tf.keras.callbacks.ReduceLROnPlateau(
            #     monitor='val_loss',
            #     factor=0.5,
            #     patience=2,
            #     min_lr=1e-6,
            #     verbose=1
            # ),
        ]

    # ------------------- Training and Evaluation on Base Dataset -------------------
    
    print("Training model...")

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    model.save(os.path.join(output_dir, 'model_final.keras'))
        
    print("Evaluating model on test dataset...")
    evaluate_and_log_model(model, output_dir, config.dataset_artifact, test_ds, None, config, class_names, validation_steps)

    # ------------------- Evaluation on Real Datasets -------------------
    test_edges_label =                      "test_edges:v0"
    test_good_condition_label =             "test_good_conditions:v0"
    test_other_objects_interference_label = "test_other_objects_interference:v0"
    test_partial_occlusion_label =          "test_partial_occlusion:v0"

    test_edges_dir =                        os.path.join(base_dir, "artifacts", test_edges_label)
    test_good_condition_dir =               os.path.join(base_dir, "artifacts", test_good_condition_label)
    test_other_objects_interference_dir =   os.path.join(base_dir, "artifacts", test_other_objects_interference_label)
    test_partial_occlusion_dir =            os.path.join(base_dir, "artifacts", test_partial_occlusion_label)
    
    evaluate_and_log_model(model, output_dir, test_edges_label, None, test_edges_dir, config, class_names, None)
    evaluate_and_log_model(model, output_dir, test_good_condition_label, None, test_good_condition_dir, config, class_names, None)
    evaluate_and_log_model(model, output_dir, test_other_objects_interference_label, None, test_other_objects_interference_dir, config, class_names, None)
    evaluate_and_log_model(model, output_dir, test_partial_occlusion_label, None, test_partial_occlusion_dir, config, class_names, None)


    # ------------------- Finish -------------------
    
    print(f"All results saved to {output_dir}")
    wandb.finish()
    
if __name__ == "__main__":
    main()
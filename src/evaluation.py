import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from data_generator import SimpleMultiViewDataGenerator
import seaborn as sns

def evaluate_and_log_model(model, output_dir, test_dataset_label, test_ds=None, test_data_dir=None, config=None, class_names=None, validation_steps=None):
    """
    Evaluates the given model on the test dataset and logs results to WandB under unique keys
    prefixed with test_dataset_label to allow multiple evaluations in one run.

    If test_ds (test dataset) is not provided, it is created using test_data_dir and config.
    In that case, validation_steps will be computed automatically from the test generator's sample count.
    """
    # If test_ds is not supplied, create the generator.
    if test_ds is None:
        if test_data_dir is None:
            raise ValueError("Either test_ds or test_data_dir must be provided.")
        views = config.get("views", ["back_left", "back_right", "front_left", "front_right", "top"]) if config else ["back_left", "back_right", "front_left", "front_right", "top"]
        input_shape = config.get("input_shape", (224, 224, 3)) if config else (224, 224, 3)
        batch_size = config.get("batch_size", 8) if config else 8

        test_gen = SimpleMultiViewDataGenerator(
            data_dir=test_data_dir,
            views=views,
            input_shape=input_shape,
            batch_size=batch_size,
            test_split=1.0
        )
        test_ds = test_gen.get_test_dataset()
        if class_names is None:
            class_names = test_gen.get_class_names()
        # Automatically compute validation_steps from the generator sample count if not provided.
        if validation_steps is None:
            validation_steps = len(test_gen.test_samples) // batch_size
            if len(test_gen.test_samples) % batch_size != 0:
                validation_steps += 1

    # Evaluate the model.
    report, cm_fig, y_true, y_pred = evaluate_model(model, test_ds, class_names, validation_steps)
    predictions_with_histogram = plot_predictions(model, test_ds, class_names)

    # Log unique keys using the test_dataset_label.
    wandb.log({
        f"{test_dataset_label}/classification_report": format_classification_report(report, class_names),
        f"{test_dataset_label}/confusion_matrix": wandb.Image(cm_fig),
        f"{test_dataset_label}/predictions_with_histogram": wandb.Image(predictions_with_histogram),
    })

def plot_confusion_matrix_from_cm(cm, class_names):
    """
    Plot a confusion matrix using seaborn with modifications
    to improve readability for a large number of classes.
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names,
                ax=ax,
                annot_kws={"fontsize": 6})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    return fig

def evaluate_model(model, test_dataset, class_names, validation_steps):
    """Evaluate model on the test dataset."""
    y_true = []
    y_pred = []
    for views, labels in test_dataset.take(validation_steps):
        batch_preds = model.predict(views, verbose=0)
        batch_pred_classes = np.argmax(batch_preds, axis=1)
        batch_true_classes = np.argmax(labels.numpy(), axis=1)
        y_true.extend(batch_true_classes)
        y_pred.extend(batch_pred_classes)
    all_labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, target_names=class_names, labels=all_labels, zero_division=0)
    print("Classification Report:")
    print(report)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_fig = plot_confusion_matrix_from_cm(cm, class_names)
    return report, cm_fig, y_true, y_pred

def format_classification_report(report, class_names):
    """
    Given a classification report string and list of class names,
    return a wandb.Table that can be logged.
    """
    report_lines = report.splitlines()
    table_data = []
    columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
    start = 2
    for line in report_lines[start:start + len(class_names)]:
        parts = line.split()
        if len(parts) < 5:
            class_name = " ".join(parts[:-4])
            parts = [class_name] + parts[-4:]
        table_data.append(parts)
    for line in report_lines[start + len(class_names):]:
        if not line.strip():
            continue
        if line.strip().startswith("accuracy"):
            parts = line.split()
            if len(parts) == 3:
                table_data.append(["accuracy", "", "", parts[1], parts[2]])
            else:
                table_data.append(parts)
        elif line.strip().startswith("macro avg") or line.strip().startswith("weighted avg"):
            parts = line.split()
            if len(parts) == 6:
                parts = [" ".join(parts[:2])] + parts[2:]
            table_data.append(parts)
    return wandb.Table(data=table_data, columns=columns)

def plot_predictions(model, test_dataset, class_names, num_samples=10, visualize_original=True):
    """
    Generate a figure to visualize predictions from the test dataset.
    For each sample (a 5-view image), the figure displays:
      - The 5 views (each in a fixed square cell).
      - A bar plot (last column) showing the softmax probability distribution over all classes,
        allocated extra horizontal space so that 80 classes can be read.
    The first view includes an annotation of the true and predicted labels.
    
    Returns:
         fig: The high-resolution matplotlib figure.
    """
    import matplotlib.gridspec as gridspec

    # Take one batch from the test dataset.
    for views, labels in test_dataset.take(1):
        preds = model.predict(views, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(labels.numpy(), axis=1)
        confidences = np.max(preds, axis=1)

        num_samples = min(num_samples, len(true_classes))
        num_views = len(views)

        # Use a fixed cell size for view images (square cells)...
        cell_size = 5  # each view cell is 5-inch square
        hist_ratio = 3  # histogram column will be 3 times as wide as a view cell

        fig_width = (num_views * cell_size) + (hist_ratio * cell_size)
        fig_height = num_samples * cell_size

        # Create a high-resolution figure.
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        # Use gridSpec: first num_views columns (ratio=1 each) for views and one column (ratio=hist_ratio) for histogram.
        gs = gridspec.GridSpec(num_samples, num_views + 1, figure=fig,
                                 width_ratios=[1]*num_views + [hist_ratio])
    
        for i in range(num_samples):
            # Plot the multi-view images.
            for v in range(num_views):
                ax = fig.add_subplot(gs[i, v])
                image = views[v][i].numpy()
                image = np.clip(image, 0, 255).astype('uint8')
                ax.imshow(image)
                ax.axis("off")
                if v == 0:
                    true_label = class_names[true_classes[i]]
                    pred_label = class_names[pred_classes[i]]
                    conf = confidences[i] * 100
                    color = "green" if true_classes[i] == pred_classes[i] else "red"
                    ax.set_title(f"T: {true_label}\nP: {pred_label}\nConf: {conf:.1f}%", color=color, fontsize=10)

            # Plot the histogram for softmax scores.
            ax_hist = fig.add_subplot(gs[i, -1])
            sample_probs = preds[i]  # softmax probabilities for sample i
            bars = ax_hist.bar(range(len(class_names)), sample_probs, color="grey", width=0.8)
            # Highlight the predicted class.
            bars[pred_classes[i]].set_color("blue")
            # Annotate each bar with its probability.
            for j, prob in enumerate(sample_probs):
                ax_hist.text(j, prob, f"{prob:.2f}", ha="center", va="bottom", fontsize=4, rotation=90)
            # Set all tick labels so they align with the bars.
            ax_hist.set_xticks(range(len(class_names)))
            ax_hist.set_xticklabels(class_names, rotation=90, fontsize=4)
            ax_hist.set_ylim([0, 1])
            ax_hist.set_title("Softmax Scores", fontsize=10)
        plt.tight_layout()
        return fig
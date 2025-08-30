#!/usr/bin/env python3
"""
eval_only.py — Evaluate saved multi-view models on synthetic + real splits.

Usage example:
python eval_only.py \
  --models_root models_dump/freezing_tests \
  --which both \
  --synthetic_dir /data/synth_test \
  --real_good_dir /data/real_good \
  --real_edges_dir /data/real_edges \
  --real_interference_dir /data/real_interference \
  --real_occlusion_dir /data/real_occlusion \
  --input_shape 224 224 3 \
  --batch_size 16 \
  --views 5 \
  --out_dir results_eval
"""
import os, sys, argparse
from pathlib import Path
import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

# --- Import your project code
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if SRC_DIR.exists():
    sys.path.append(str(SRC_DIR))
from data_generator import SimpleMultiViewDataGenerator  # type: ignore

@keras.saving.register_keras_serializable()
class StackReduceLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(StackReduceLayer, self).__init__(**kwargs)
    def call(self, inputs):
        stacked = tf.stack(inputs, axis=1)
        return tf.reduce_max(stacked, axis=1)
    def get_config(self):
        base_config = super(StackReduceLayer, self).get_config()
        return base_config

# If you used custom layers, register them for load_model (optional).
try:
    CUSTOM_OBJECTS = {"StackReduceLayer": StackReduceLayer}
except Exception:
    CUSTOM_OBJECTS = {}

print("Custom objects for loading:", CUSTOM_OBJECTS)

def tf_configure():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

def load_keras_model(model_path: Path):
    print(f"[load] {model_path}")
    return tf.keras.models.load_model(str(model_path), custom_objects=CUSTOM_OBJECTS)

def build_test_ds_from_dir(data_dir: Path, input_shape, batch_size, views):
    gen = SimpleMultiViewDataGenerator(
        data_dir=str(data_dir),
        input_shape=tuple(input_shape),
        batch_size=batch_size,
        random_state=42,
        preprocess_fn=None

    )

    return gen.get_test_dataset(), gen.get_class_names(), None

def eval_on_dataset(model, ds, class_names, split_name: str, out_dir: Path):
    y_true, y_pred = [], []
    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(batch_y.numpy(), axis=1))
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    labels = list(range(len(class_names)))

    # Aggregates
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    # Per-class report
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        zero_division=0, output_dict=True
    )
    rep_df = pd.DataFrame(report).T
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(out_dir / f"classification_report__{split_name}.csv", float_format="%.4f")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_title(f"Confusion Matrix — {split_name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix__{split_name}.png", dpi=200)
    plt.close(fig)

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

def find_model_paths(models_root: Path, which: str):
    """
    Yield (experiment_name, checkpoint_name, model_path)
    """
    out = []
    for exp_dir in sorted(models_root.glob("*")):
        if not exp_dir.is_dir(): continue
        for ck in ("best_epoch","last_epoch"):
            if which in ("both", ck, "best" if ck=="best_epoch" else "last"):
                mp = exp_dir / ck / "model_best.keras"
                if mp.exists():
                    out.append((exp_dir.name, ck, mp))
    return out

def main():
    tf_configure()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", type=str, required=True)
    ap.add_argument("--which", choices=["best","last","both"], default="best")
    ap.add_argument("--synthetic_dir", type=str)
    ap.add_argument("--real_good_dir", type=str)
    ap.add_argument("--real_edges_dir", type=str)
    ap.add_argument("--real_interference_dir", type=str)
    ap.add_argument("--real_occlusion_dir", type=str)
    ap.add_argument("--input_shape", nargs=3, type=int, default=[224,224,3])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="results_eval")
    args = ap.parse_args()

    models_root = Path(args.models_root)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    # Which datasets to evaluate
    ds_specs = []
    if args.synthetic_dir: ds_specs.append(("synthetic", Path(args.synthetic_dir)))
    for name, p in [("real_good", args.real_good_dir),
                    ("real_edges", args.real_edges_dir),
                    ("real_interference", args.real_interference_dir),
                    ("real_occlusion", args.real_occlusion_dir)]:
        if p: ds_specs.append((name, Path(p)))

    model_entries = find_model_paths(models_root, args.which)
    if not model_entries:
        print("No models found under", models_root); sys.exit(1)

    rows = []
    for exp_name, ck_name, model_path in model_entries:
        print(f"\n=== {exp_name} [{ck_name}] ===")
        model = load_keras_model(model_path)
        exp_out = out_root / exp_name / ck_name; exp_out.mkdir(parents=True, exist_ok=True)

        for split_name, split_dir in ds_specs:
            print(f"[dataset] {split_name}: {split_dir}")
            ds, class_names, _ = build_test_ds_from_dir(split_dir, args.input_shape, args.batch_size, args.views)
            metrics = eval_on_dataset(model, ds, class_names, split_name, exp_out)
            rows.append({"experiment": exp_name, "checkpoint": ck_name, "split": split_name, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "summary.csv", index=False, float_format="%.4f")
    print("\nSummary:\n", df)

if __name__ == "__main__":
    main()

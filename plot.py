import re
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

def parse_train_log(log_path):
    metrics = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "error_rate": [],
        "micro_f1": [],
        "macro_f1": [],
        "precision": [],
        "recall": []
    }

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(
                r"Epoch: (\d+).*?train loss: ([\d\.e\-]+).*?valid loss: ([\d\.e\-]+), valid error_rate: ([\d\.e\-]+), "
                r"valid micro_f1: ([\d\.e\-]+), valid macro_f1: ([\d\.e\-]+), "
                r"valid mean_precision: ([\d\.e\-]+), valid mean_recall: ([\d\.e\-]+)", line)

            if match:
                metrics["epoch"].append(int(match.group(1)))
                metrics["train_loss"].append(float(match.group(2)))
                metrics["valid_loss"].append(float(match.group(3)))
                metrics["error_rate"].append(float(match.group(4)))
                metrics["micro_f1"].append(float(match.group(5)))
                metrics["macro_f1"].append(float(match.group(6)))
                metrics["precision"].append(float(match.group(7)))
                metrics["recall"].append(float(match.group(8)))

    return metrics


def plot_metrics(metrics, output_folder="plots"):
    import os
    os.makedirs(output_folder, exist_ok=True)

    def plot_single(metric_name, ylabel=None):
        plt.figure()
        plt.plot(metrics["epoch"], metrics[metric_name])
        plt.xlabel("Epoch")
        plt.ylabel(ylabel or metric_name)
        plt.title(metric_name.replace("_", " ").title())
        plt.grid(True)
        plt.savefig(f"{output_folder}/{metric_name}.png")
        plt.close()

    plot_single("train_loss", "Train Loss")
    plot_single("valid_loss", "Validation Loss")
    plot_single("error_rate", "Error Rate")
    plot_single("micro_f1", "Micro F1 Score")
    plot_single("macro_f1", "Macro F1 Score")
    plot_single("precision", "Mean Precision")
    plot_single("recall", "Mean Recall")

    plt.figure()
    plt.plot(metrics["epoch"], metrics["macro_f1"], label="Macro F1")
    plt.plot(metrics["epoch"], metrics["precision"], label="Mean Precision")
    plt.plot(metrics["epoch"], metrics["recall"], label="Mean Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_folder}/combined_metrics.png")
    plt.close()

#metrics = parse_train_log("results/9999/train_log.txt")
#plot_metrics(metrics, "results/9999/plots")

def confusion_matrix_plot(input_json, output_folder="plots", normalized="true"):
    with open(input_json, "r") as f:
        data = json.load(f)

    y_true = [entry["true_label"] for entry in data.values()]
    y_pred = [entry["predicted_class"] for entry in data.values()]

    le = LabelEncoder()
    le.fit(y_true + y_pred)
    labels = le.classes_


    if normalized == "true":
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format=".2f")
        plt.title("Normalized Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/normalized_confusion_matrix.png")
        plt.close()
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/confusion_matrix.png")
        plt.close()

confusion_matrix_plot("results/9999/labels.json", "results/9999/plots/", "false")
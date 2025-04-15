import numpy as np
import torch

from speechbrain.utils.metric_stats import MetricStats

class MultiClassClassificationMetricStats(MetricStats):
    def __init__(self):
        super()
        self.clear()
        self.summary = None

    def append(self, ids, predictions, targets):
        """
        Appends inputs, predictions and targets to internal
        lists

        Arguments
        ---------
        ids: list
            the string IDs for the samples
        predictions: list
            the model's predictions (human-interpretable,
            preferably strings)
        targets: list
            the ground truths (human-interpretable, preferably strings)
        categories: list
            an additional way to classify training
            samples. If available, the categories will
            be combined with targets
        """
        self.ids.extend(ids)
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def summarize(self, field=None):
        self.n = np.max(self.targets) + 1
        
        accuracy = self._compute_accuracy()
        confusion_matrix = self._compute_confusion_matrix()
        
#        print(confusion_matrix)
        
        precisions = self._compute_precision(confusion_matrix)
        recalls = self._compute_recall(confusion_matrix)
        f1s = self._compute_f1(precisions, recalls)
        self.summary = {
            "accuracy": accuracy,
            "error_rate": 1 - accuracy,
            "confusion_matrix": confusion_matrix,
            "mean_precision" : np.mean(precisions),
            "mean_recall" : np.mean(recalls),
            "micro_f1" : self._compute_micro_f1(f1s, confusion_matrix),
            "macro_f1" : np.mean(f1s),
#            "precisions" : precisions,
#            "recalls" : recalls,
#            "f1s" : f1s
        }
        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def _compute_accuracy(self):
        return sum(
            prediction == target
            for prediction, target in zip(self.predictions, self.targets)
        ) / len(self.ids)
        
    def _compute_precision(self, confusion_matrix):
        N_predicted = np.sum(confusion_matrix, axis=0)
        precisions = np.zeros(self.n, dtype=np.float32)
        
        for class_idx in range(0, self.n):
            precisions[class_idx] = np.float32(confusion_matrix[class_idx][class_idx]) / N_predicted[class_idx]
        precisions = np.nan_to_num(precisions)
        return precisions

    def _compute_recall(self, confusion_matrix):
        N_target = np.sum(confusion_matrix, axis=1)
        recalls = np.zeros(self.n, dtype=np.float32)
        
        for class_idx in range(0, self.n):
            recalls[class_idx] = np.float32(confusion_matrix[class_idx][class_idx]) / N_target[class_idx]
        recalls = np.nan_to_num(recalls)
        return recalls
        
    def _compute_f1(self, precisions, recalls):
        f1 = np.zeros(self.n, dtype=np.float32)
        for class_idx in range(0, self.n):
            f1[class_idx] = 2 * precisions[class_idx] * recalls[class_idx] / (precisions[class_idx] + recalls[class_idx])
        f1 = np.nan_to_num(f1)
        return f1    

    def _compute_micro_f1(self, f1s, confusion_matrix):
        N = np.sum(confusion_matrix)
        micro_f1 = 0.0
        
        for class_idx in range(0, self.n):
            micro_f1 += np.float32(confusion_matrix[class_idx][class_idx]) / N * f1s[class_idx]
        micro_f1 = np.nan_to_num(micro_f1)
        return micro_f1    

    def _compute_confusion_matrix(self):
#        confusion_matrix = torch.zeros(self.n, self.n)
        confusion_matrix = np.zeros((self.n, self.n), dtype=np.int32)
        for key, prediction in self._get_confusion_entries():
            confusion_matrix[key, prediction] += 1
        return confusion_matrix

    def _compute_classwise_stats(self, confusion_matrix):
        total = confusion_matrix.sum(dim=-1)
        
        correct = 0
        for idx in range(0, self.n):
            correct += confusion_matrix[idx][idx]
        accuracy = correct / total
        
        return {
            key: {
                "total": item_total.item(),
                "correct": item_correct.item(),
                "accuracy": item_accuracy.item(),
            }
            for key, item_total, item_correct, item_accuracy in zip(
                self._available_keys, total, correct, accuracy
            )
        }

    def _get_keys(self):
        keys = self.targets
        return list(sorted(set(keys)))

    def _get_confusion_entries(self):
        result = zip(self.targets, self.predictions)
        result = list(result)
        return result

    def clear(self):
        """Clears the collected statistics"""
        self.ids = []
        self.predictions = []
        self.targets = []

    def write_stats(self, filestream):
        """Outputs the stats to the specified filestream in a human-readable format

        Arguments
        ---------
        filestream: file
            a file-like object
        """
        if self.summary is None:
            self.summarize()
        print(
            f"Overall Accuracy: {self.summary['accuracy']:.0%}", file=filestream
        )
        print(file=filestream)
        self._write_classwise_stats(filestream)
        print(file=filestream)
        self._write_confusion(filestream)

    def _write_classwise_stats(self, filestream):
        self._write_header("Class-Wise Accuracy", filestream=filestream)
        key_labels = {
            key: self._format_key_label(key) for key in self._available_keys
        }
        longest_key_label = max(len(label) for label in key_labels.values())
        for key in self._available_keys:
            stats = self.summary["classwise_stats"][key]
            padded_label = self._pad_to_length(
                self._format_key_label(key), longest_key_label
            )
            print(
                f"{padded_label}: {int(stats['correct'])} / {int(stats['total'])} ({stats['accuracy']:.2%})",
                file=filestream,
            )

    def _write_confusion(self, filestream):
        self._write_header("Confusion", filestream=filestream)
        longest_prediction = max(
            len(prediction) for prediction in self._available_predictions
        )
        confusion_matrix = self.summary["confusion_matrix"].int()
        totals = confusion_matrix.sum(dim=-1)
        for key, key_predictions, total in zip(
            self._available_keys, confusion_matrix, totals
        ):
            target_label = self._format_key_label(key)
            print(f"Target: {target_label}", file=filestream)
            (indexes,) = torch.where(key_predictions > 0)
            total = total.item()
            for index in indexes:
                count = key_predictions[index].item()
                prediction = self._available_predictions[index]
                padded_label = self._pad_to_length(
                    prediction, longest_prediction
                )
                print(
                    f"  -> {padded_label}: {count} / {total} ({count / total:.2%})",
                    file=filestream,
                )

    def _write_header(self, header, filestream):
        print(header, file=filestream)
        print("-" * len(header), file=filestream)

    def _pad_to_length(self, label, length):
        padding = max(0, length - len(label))
        return label + (" " * padding)

    def _format_key_label(self, key):
        return key



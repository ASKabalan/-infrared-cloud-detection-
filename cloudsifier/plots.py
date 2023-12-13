# -*- coding: utf-8 -*-
# pylint: disable=R0913

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

DPI = 300


def matrix_confusion(y_true, y_pred, plotsdir, case, title):
    class_names = ["Negative", "Positive"]
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, numpy.newaxis]
    seaborn.heatmap(cm_normalized, annot=True, fmt=".3", cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot_kws={"fontsize": 15})
    pyplot.xlabel("Predicted")
    pyplot.ylabel("True")
    pyplot.title(title)
    pyplot.show()
    pyplot.savefig(plotsdir / f"{case}_confusion_matrix.pdf", dpi=DPI)
    pyplot.close()


def roc(preds, truth, plotsdir, case):
    fpr, tpr, _ = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)
    pyplot.figure(figsize=(6, 4), dpi=DPI)
    pyplot.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc})")
    pyplot.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("Receiver Operating Characteristic (ROC) Curve")
    pyplot.legend(loc="lower right")
    pyplot.grid(alpha=0.75, ls="dashed")
    pyplot.savefig(plotsdir / f"{case}_roc_plot.pdf", dpi=DPI)
    pyplot.close()


def loss_and_accuracy(loss, val_loss, acc, val_acc, plotsdir, case):
    nb_epochs = numpy.arange(1, len(loss) + 1)
    fig, ax1 = pyplot.subplots(figsize=(10, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Log Loss", color="tab:red")
    ax1.plot(nb_epochs, numpy.log(loss), label="Training Loss", color="tab:red")
    ax1.plot(nb_epochs, numpy.log(val_loss), label="Validation Loss", color="tab:orange", linestyle="dashed")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    pyplot.savefig(plotsdir / f"{case}_log_losses.pdf", dpi=DPI)
    pyplot.close()

    fig, ax1 = pyplot.subplots(figsize=(10, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(nb_epochs, loss, label="Training Loss", color="tab:red")
    ax1.plot(nb_epochs, val_loss, label="Validation Loss", color="tab:orange", linestyle="dashed")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    pyplot.savefig(plotsdir / f"{case}_losses.pdf", dpi=DPI)
    pyplot.close()

    fig, ax1 = pyplot.subplots(figsize=(10, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color="tab:red")
    ax1.plot(nb_epochs, acc, label="Training Acc", color="tab:red")
    ax1.plot(nb_epochs, val_acc, label="Validation Acc", color="tab:orange", linestyle="dashed")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    pyplot.savefig(plotsdir / f"{case}_acc.pdf", dpi=DPI)
    pyplot.close()


def report(preds, truth, plotsdir, case):
    res = pandas.DataFrame(classification_report(truth, preds, output_dict=True))[["macro avg", "accuracy"]]
    res.to_csv(plotsdir / f"{case}_report.csv", index=False)

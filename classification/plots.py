# -*- coding: utf-8 -*-
# pylint: disable=R0913

import numpy
import seaborn
from matplotlib import pyplot
from sklearn.metrics import auc, confusion_matrix, roc_curve


def matrix_confusion(y_true, y_pred, plotsdir, case, title):
    class_names = ["Negative", "Positive"]
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, numpy.newaxis]
    seaborn.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot_kws={"fontsize": 15})
    pyplot.xlabel("Predicted")
    pyplot.ylabel("True")
    pyplot.title(title)
    pyplot.savefig(plotsdir / f"{case}_confusion_matrix.pdf", dpi=300)
    pyplot.close()


def roc(preds, truth, plotsdir, case):
    fpr, tpr, _ = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)
    pyplot.figure(figsize=(6, 4), dpi=300)
    pyplot.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc})")
    pyplot.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("Receiver Operating Characteristic (ROC) Curve")
    pyplot.legend(loc="lower right")
    pyplot.grid(alpha=0.75, ls="dashed")
    pyplot.savefig(plotsdir / f"{case}_roc_plot.pdf", dpi=300)
    pyplot.close()


def curves_loss_and_accuracy(loss, val_loss, acc, val_acc, plotsdir, case):
    nb_epochs = numpy.arange(1, len(loss) + 1)
    fig, ax1 = pyplot.subplots(figsize=(10, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(nb_epochs, loss, label="Training Loss", color="tab:red")
    ax1.plot(nb_epochs, val_loss, label="Validation Loss", color="tab:orange", linestyle="dashed")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(nb_epochs, acc, label="Training Accuracy", color="tab:blue")
    ax2.plot(nb_epochs, val_acc, label="Validation Accuracy", color="tab:cyan", linestyle="dashed")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    pyplot.savefig(plotsdir / f"{case}_loss_and_accuracy.pdf", dpi=300)
    pyplot.close()
